import datasets
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from dataset import EuroSATDataset
from torchvision.models import ResNet50_Weights, resnet50

# Constants retrieved from:
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
RESNET_50_WEIGHT_MEAN = [0.485, 0.456, 0.406]
RESNET_50_WEIGHT_STD = [0.229, 0.224, 0.225]


def build_dataloaders(
    batch_size: int,
    num_workers: int,
):
    eurosat_rgb = datasets.load_dataset("blanchon/EuroSAT_RGB")
    num_classes = eurosat_rgb["train"].features["label"].num_classes

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(RESNET_50_WEIGHT_MEAN, RESNET_50_WEIGHT_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(RESNET_50_WEIGHT_MEAN, RESNET_50_WEIGHT_STD),
        ]
    )

    train_ds = EuroSATDataset(eurosat_rgb["train"], train_tf)
    val_ds = EuroSATDataset(eurosat_rgb["validation"], eval_tf)

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

    return train_loader, val_loader, num_classes


# Helper function to get the device CPU or GPU available to train the models.
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, device: torch.device, pretrained: bool) -> nn.Module:
    if pretrained:
        model = resnet50(weight=ResNet50_Weights.IMAGENET1K_V1)
    else: 
        model = resnet50(weights=None)
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
    
        train_loss = total_loss / max(n, 1) # max(n, 1) to avoid division by zero
    return train_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
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
