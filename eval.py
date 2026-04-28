import argparse
import json
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)
from train import build_dataloaders, build_model, get_device


MSI_BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]


def create_eval_run_dir(base_outdir: str, modality: str, ckpt_path: str) -> Path:
    base_dir = Path(base_outdir)
    base_dir.mkdir(parents=True, exist_ok=True)

    ckpt_stem = Path(ckpt_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{modality}_{ckpt_stem}_{timestamp}"
    run_dir = base_dir / run_name

    # Very unlikely, but avoid collisions if multiple evals start in same second.
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{run_name}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@torch.no_grad()
def compute_msi_band_importance(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    baseline_macro_f1: float,
):
    model.eval()
    rows = []

    for b, band_name in enumerate(MSI_BAND_NAMES):
        y_true_b, y_pred_b = [], []

        for images, labels in loader:
            images = images.to(device)

            x = images.clone()
            x[:, b, :, :] = 0.0

            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()

            y_pred_b.append(preds)
            y_true_b.append(labels.numpy())

        y_true_b = np.concatenate(y_true_b)
        y_pred_b = np.concatenate(y_pred_b)
        macro_f1_b = float(f1_score(y_true_b, y_pred_b, average="macro"))

        rows.append(
            {
                "band_index": b,
                "band_name": band_name,
                "macro_f1_masked": macro_f1_b,
                "importance": baseline_macro_f1
                - macro_f1_b,  # bigger drop = more important
            }
        )

    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_prob.append(probs)
        y_pred.append(preds)
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["rgb", "msi"], required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="eval_results")
    args = p.parse_args()

    run_dir = create_eval_run_dir(
        base_outdir=args.outdir, modality=args.modality, ckpt_path=args.ckpt
    )
    device = get_device()

    train_loader, val_loader, num_classes, in_channels = build_dataloaders(
        modality=args.modality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = val_loader

    model = build_model(num_classes=num_classes, device=device, in_channels=in_channels)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred, y_prob = collect_predictions(model, loader, device)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    if args.modality == "msi":
        band_rows = compute_msi_band_importance(
            model, loader, device, baseline_macro_f1=metrics["macro_f1"]
        )

        with open(run_dir / "band_importance.json", "w") as f:
            json.dump(band_rows, f, indent=2)
        
        with open(run_dir / "band_importance.csv", "w") as f:
            f.write("band_index,band_name,macro_f1_masked,importance\n")
            for r in band_rows:
                f.write(
                    f'{r["band_index"]},{r["band_name"]},'
                    f'{r["macro_f1_masked"]:.6f},{r["importance"]:.6f}\n'
                )

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savetxt(run_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    np.savetxt(run_dir / "confusion_matrix_normalized.csv", cm_norm, delimiter=",")

    report = classification_report(y_true, y_pred, output_dict=True)
    with open(run_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    latest_run_file = Path(args.outdir) / "latest_run.txt"
    latest_run_file.write_text(f"{run_dir}\n", encoding="utf-8")
    print(f"Saved evaluation artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
