from pathlib import Path

import torch
from PIL import Image

from train import build_model, build_rgb_transform


CLASS_NAMES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial Buildings",
    "Pasture",
    "Permanent Crop",
    "Residential Buildings",
    "River",
    "SeaLake",
]

DEFAULT_CHECKPOINT_PATH = Path("weights/rgb_e15_best.pt")


def load_rgb_model(checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH) -> torch.nn.Module:
    """Load the EuroSAT-RGB ResNet-50 checkpoint for CPU inference."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"RGB checkpoint not found at {checkpoint_path}. "
            "Add weights/rgb_e15_best.pt before running the demo."
        )

    device = torch.device("cpu")
    model = build_model(num_classes=len(CLASS_NAMES), device=device, in_channels=3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_topk(
    model: torch.nn.Module, image: Image.Image, top_k: int = 3
) -> list[tuple[str, float]]:
    """Run RGB inference and return class names with probabilities."""
    transform = build_rgb_transform(train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    top_probs, top_indices = torch.topk(probs, k=top_k)
    return [
        (CLASS_NAMES[int(class_idx)], float(prob))
        for prob, class_idx in zip(top_probs, top_indices, strict=True)
    ]
