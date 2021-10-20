import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision
from typing import Callable


class ImageNetForegroundModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "",
        batch_size: int = 1,
        num_workers=16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()

    def preprocessing(self) -> Callable:
        steps = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ]
        )
        return steps

    def setup(self):
        train_data_dir = os.path.join(self.data_dir, "train")
        self.train_dataset = torchvision.datasets.ImageFolder(
            train_data_dir, transform=self.preprocessing()
        )
        self.class_to_idx = self.train_dataset.class_to_idx
        self.idx_to_class = ImageNetForegroundModule.reverse_lookup(
            self.train_dataset.class_to_idx
        )

    @staticmethod
    def reverse_lookup(lookup: dict):
        """Reverses dictionary lookup"""
        inv_map = {v: k for k, v in lookup.items()}
        return inv_map

    def train_dataloader(self) -> DataLoader:
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        return data_loader
