from train import get_device
import datasets

device = get_device()
print(device)

eurosat_rgb = datasets.load_dataset("blanchon/EuroSAT_RGB")
num_classes = eurosat_rgb["train"].features["label"].num_classes


print(num_classes)