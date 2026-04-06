from torch.utils.data import Dataset

class EuroSATDataset(Dataset):
    def __init__(self, split, transform):
        self._data = split
        self.transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data[idx]
        image = row["image"]
        label = row["label"]
        return self.transform(image), label