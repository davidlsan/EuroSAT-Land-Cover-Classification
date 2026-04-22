import os
import random
from torch import nn
import torch
from cli import parse_args
from train import build_dataloaders, build_model, evaluate, get_device, train_one_epoch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs("weights", exist_ok=True)

    # change 32 -> 16 for lower end computers
    train_loader, val_loader, num_classes, in_channels = build_dataloaders(
        modality=cfg.modality, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    modality = cfg.modality
    ckpt_path = f"weights/{modality}_best.pt"

    device = get_device()
    model = build_model(num_classes=num_classes, device=device, in_channels=in_channels)
    print(f"Training using {modality} dataset")

    criterion = nn.CrossEntropyLoss()

    # Recommended to use --lr 1e-3 when running from CLI (Default Adam number)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # LR Scheduler to decay LR over time (due to training run val_acc spikes)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0
    print(f"Training on {device}")

    # Train for N epochs (decided by CLI param: --epochs N), recommended 15
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Acc: {val_acc}"
        )

        scheduler.step()
        print(f" Current LR value {optimizer.param_groups[0]['lr']}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "modality": cfg.modality,
                },
                ckpt_path,
            )
            print(f"Saved new best model with val_acc {best_val_acc}")


if __name__ == "__main__":
    main()
