from pathlib import Path
from typing import Type

import lightning as L
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from src.dataloaders.transforms import TestTransforms
from src.utils.data_utils import prepare_corn_data
from src.utils.read_write import read_json

# TEST_DATA_DIR = "CORN-3_noD"
TEST_DATA_DIR = "CORN-3_orig"
FILE_EXTENSION = ".tif"
DATA_SPLIT_FILE = "data_split.json"

class CNFDataModule(L.LightningDataModule):

    def __init__(
            self,
            data_dir: Path,
            batch_size: int,
            num_workers: int,
            dataset: Type[Dataset],
            val_size: float = 0.3,
            train_transforms=None,
            test_transforms=TestTransforms(),
            file_extension: str = ".tif",
    ):
        super().__init__()

        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        self.test_image_paths = []
        self.test_labels = []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.file_extension = file_extension
        self.dataset = dataset

        self.data_dict = {}

    def prepare_data(self):
        # download only
        # single gpu
        self.data_split = read_json(self.data_dir.parent / DATA_SPLIT_FILE)
        self.train_files = [Path(train_file_dir) for train_file_dir in self.data_split["train_files"]]
        self.train_labels = self.data_split["train_labels"]
        self.val_files = [Path(val_file_dir) for val_file_dir in self.data_split["val_files"]]
        self.val_labels = self.data_split["val_labels"]
        self.test_image_paths, self.test_labels = prepare_corn_data(self.data_dir.parent / TEST_DATA_DIR, FILE_EXTENSION)

    def setup(self, stage):
        self.train_dataset = self.dataset(self.train_files, self.train_labels, transform=self.train_transforms.compose())
        self.val_dataset = self.dataset(self.val_files, self.val_labels, transform=self.test_transforms.compose())
        self.test_dataset = self.dataset(self.test_image_paths, self.test_labels, transform=self.test_transforms.compose())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers = True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def train_val_dataloader(self):
        return DataLoader(
            ConcatDataset([self.train_dataset, self.val_dataset]),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
