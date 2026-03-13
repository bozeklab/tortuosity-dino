from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CNFDataset(Dataset):

    def __init__(self, image_paths: list[Path], labels=list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample_path = self.image_paths[idx]
        image = Image.open(sample_path).convert("L")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
