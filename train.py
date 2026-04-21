import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torchvision
from tqdm import tqdm
from dataset import EuroSATDataset
import torch.nn.functional as F

# Constants retrieved from:
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
RESNET_50_WEIGHT_MEAN = [0.485, 0.456, 0.406]
RESNET_50_WEIGHT_STD = [0.229, 0.224, 0.225]

DATASET_CFG = {
    "rgb": {"hf_id": "blanchon/EuroSAT_RGB", "in_channels": 3},
    "msi": {"hf_id": "blanchon/EuroSAT_MSI", "in_channels": 13},
}


def to_chw_tensor(image):
    hwc = np.array(image, dtype=np.float32)  # HWC typical shape: 64x64x3
    chw = torch.from_numpy(hwc).permute(2, 0, 1)  # CHW typical shape: 3x64x64
    return chw


def build_rgb_transform(train: bool):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(RESNET_50_WEIGHT_MEAN, RESNET_50_WEIGHT_STD),
        ]
    )
    return transforms.Compose(ops)


def build_msi_transform(train: bool):
    def _tf(image):
        chw = to_chw_tensor(image)
        chw = chw / 10000.0
        if train and torch.rand(1).item() < 0.5:
            chw = torch.flip(chw, dims=[2])
        chw = F.interpolate(
            chw.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)
        return chw

    return _tf


def build_dataloaders(
    modality: str,
    batch_size: int,
    num_workers: int,
):
    cfg = DATASET_CFG[modality]
    ds = datasets.load_dataset(cfg["hf_id"])
    in_channels = cfg["in_channels"]
    num_classes = ds["train"].features["label"].num_classes

    if modality == "rgb":
        train_tf = build_rgb_transform(train=True)
        eval_tf = build_rgb_transform(train=False)
    else:
        train_tf = build_msi_transform(train=True)
        eval_tf = build_msi_transform(train=False)

    train_ds = EuroSATDataset(ds["train"], train_tf)
    val_ds = EuroSATDataset(ds["validation"], eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, num_classes, in_channels


# Helper function to get the device CPU or GPU available to train the models.
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, device: torch.device, in_channels: int) -> nn.Module:
    model = torchvision.models.resnet50(weights=None)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    n = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_n = labels.size(0)
        total_loss += loss.item() * batch_n
        n += batch_n

        train_loss = total_loss / max(n, 1)  # max(n, 1) to avoid division by zero
    return train_loss


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        val_loss = total_loss / total
        val_acc = correct / total

    return val_loss, val_acc
