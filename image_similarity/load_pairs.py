"""
Loads pairs across- or within- image classes
"""
from pathlib import Path
from typing import List
import torchvision
import warnings
import torch
import os


def load_dataset(
    image_net_dir: Path, class_label: str, randomly_rotate: bool = False
) -> torch.utils.data.Dataset:
    """Returns a torch dataset from the specified class"""
    class_dir = image_net_dir / Path(f"{class_label}")
    # follows official pytorch example
    preprocessing_steps = [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]

    if randomly_rotate:
        preprocessing_steps.append(torchvision.transforms.RandomRotation([0.0, 360.0]))
    class_dataset = torchvision.datasets.ImageFolder(
        class_dir,
        torchvision.transforms.Compose(preprocessing_steps),
    )
    return class_dataset


def get_class_data_loader(
    image_net_dir: Path,
    class_label: str = "n01629819",
    batch_size: int = 1,
    randomly_rotate: bool = False,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Returns a data loader containing images from specified class"""
    class_dataset = load_dataset(
        image_net_dir, class_label, randomly_rotate=randomly_rotate
    )

    class_data_loader = torch.utils.data.DataLoader(
        class_dataset, shuffle=False, batch_size=batch_size, drop_last=drop_last
    )
    return class_data_loader


def get_across_classes_pairs_data_loader(
    image_net_dir: Path,
    class_label_1: str = "n01629819",
    class_label_2: str = "n01443537",
    randomly_rotate: bool = False,
) -> torch.utils.data.DataLoader:
    """Returns a data loader for pairs of images from specified classes"""
    class1_dataset = load_dataset(
        image_net_dir, class_label_1, randomly_rotate=randomly_rotate
    )
    class2_dataset = load_dataset(
        image_net_dir, class_label_2, randomly_rotate=randomly_rotate
    )

    loader = torch.utils.data.DataLoader(
        ConcatDataset(
            class1_dataset,
            class2_dataset,
        ),
        batch_size=1,
    )
    return loader


class ConcatDataset(torch.utils.data.Dataset):
    """Concatenates datasets into a single dataset"""

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def get_image_net_dir(use_val=True):
    subdir = "val" if use_val else "train"
    image_net_dir = Path(f"")
    if not image_net_dir.exists():
        if use_val:
            warnings.warn("using training set for tiny imagenet")
        # for testing locally
        image_net_dir = Path(
            ""
        )
    return image_net_dir


def load_class_labels(image_net_dir: Path) -> List[str]:
    labels = []
    for f in os.listdir(image_net_dir):
        if f.startswith("n"):
            labels.append(f)
    return labels


def symlink_image_net(image_net_dir: Path, dest: Path) -> None:
    """Creates a symlinked directory with additional directory for ImageFolder.
    Example: dest/[label]/[label]/img.jpeg
    """
    for class_label in os.listdir(image_net_dir):
        os.makedirs(dest / class_label)
        os.symlink(
            image_net_dir / class_label,
            dest / class_label / class_label,
            target_is_directory=True,
        )
