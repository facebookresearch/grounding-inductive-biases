"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torchvision
import os
import pytorch_lightning as pl
from per_class_augmentation import augmentations as augmentations_lib
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import transforms as transform_lib
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Callable, Any, Tuple, Optional, Dict, List


class TopAugmentationsDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        image_size: int = 224,
        transform_dir: str = None,
        num_transforms: int = 25,
        similarity_type: str = "resnet18_no_aug",
        plus_standard_aug: bool = False,
        standard_aug_before: bool = True,
        top_per_class: bool = True,
        top_transform_ranking: str = "proportion_boosted",
        transform_prob: dict = {"dist": "weighted_boost"},
        min_prop_boosted_filter: Optional[float] = 0.4,
        min_perc_change_per_class_filter: Optional[float] = 0.0,
    ):
        super().__init__(root)
        self.idx_to_class = self.reverse_lookup(self.class_to_idx)

        self.image_size = image_size
        self.plus_standard_aug = plus_standard_aug
        self.standard_aug_before = standard_aug_before

        self.preprocessing = self.get_preprocessing_steps()
        self.postprocessing = self.get_post_top_steps()
        self.to_tensor_normalization = self.to_tensor_normalization_steps()

        self.top_augmentations = augmentations_lib.TopWeightedAugmentations(
            num_transforms=num_transforms,
            transform_dir=transform_dir,
            similarity_type=similarity_type,
            transform_prob=transform_prob,
            top_transform_ranking=top_transform_ranking,
            top_per_class=top_per_class,
            min_prop_boosted_filter=min_prop_boosted_filter,
            min_perc_change_per_class_filter=min_perc_change_per_class_filter,
        )

    def get_preprocessing_steps(self) -> Callable:
        if self.plus_standard_aug:
            if self.standard_aug_before:
                return self.standard_aug_steps()
        else:
            return self.inference_aug_steps()
        return lambda x: x

    def get_post_top_steps(self) -> Callable:
        if self.plus_standard_aug:
            if not self.standard_aug_before:
                return self.standard_aug_steps()
        else:
            return self.inference_aug_steps()
        return lambda x: x

    def standard_aug_steps(self):
        steps = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
            ]
        )
        return steps

    def inference_aug_steps(self):
        steps = transform_lib.Compose(
            [
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
            ]
        )
        return steps

    def to_tensor_normalization_steps(self):
        normalization_transforms = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                imagenet_normalization(),
            ]
        )
        return normalization_transforms

    @staticmethod
    def reverse_lookup(lookup: dict):
        """Reverses dictionary lookup"""
        inv_map = {v: k for k, v in lookup.items()}
        return inv_map

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Applies per class augmentation when getting an item.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        class_label = self.idx_to_class[target]
        sample = self.loader(path)
        # standard or inference steps (before totensor)
        sample = self.preprocessing(sample)
        # apply top transform per class
        top_transforms = self.top_augmentations.get_top_transforms(
            class_label=class_label
        )
        sample = top_transforms(sample)
        sample = self.postprocessing(sample)
        # normalize and convert to tensor
        sample = self.to_tensor_normalization(sample)

        return sample, target


class ManualAugmentationDataset(ImageFolder):
    """Applies the given transforms to the specified class labels"""

    def __init__(
        self,
        root: str,
        class_label_to_transforms: Dict[str, List[Callable]],
        image_size: int = 224,
    ):
        super().__init__(root)
        self.class_label_to_transforms = class_label_to_transforms
        print("class_label to transform", self.class_label_to_transforms)
        self.idx_to_class = self.reverse_lookup(self.class_to_idx)

        self.image_size = image_size
        self.standard_aug = self.standard_aug_steps()
        self.to_tensor_normalization = self.to_tensor_normalization_steps()

    @staticmethod
    def reverse_lookup(lookup: dict):
        """Reverses dictionary lookup"""
        inv_map = {v: k for k, v in lookup.items()}
        return inv_map

    def standard_aug_steps(self):
        steps = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
            ]
        )
        return steps

    def to_tensor_normalization_steps(self):
        normalization_transforms = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                imagenet_normalization(),
            ]
        )
        return normalization_transforms

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Applies per class augmentation when getting an item.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        class_label = self.idx_to_class[target]
        sample = self.loader(path)
        # standard or inference steps (before totensor)
        sample = self.standard_aug(sample)
        class_transforms = self.class_label_to_transforms.get(class_label, [])
        for transform in class_transforms:
            sample = transform(sample)
        sample = self.to_tensor_normalization(sample)
        return sample, target


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        num_workers=16,
        image_size=224,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        augmentations = self.train_transform()
        data_loader = self._create_dataloader("train", augmentations)
        return data_loader

    def val_dataloader(self) -> DataLoader:
        augmentations = self.val_transform()
        data_loader = self._create_dataloader("val", augmentations)
        return data_loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def _create_dataloader(self, stage: str, augmentations: transform_lib.Compose):
        path = os.path.join(self.data_dir, stage)
        shuffle = True if stage == "train" else False
        dataset = torchvision.datasets.ImageFolder(path, augmentations)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
        return data_loader

    def train_transform(self) -> Callable:
        """
        The standard imagenet transforms
        """
        preprocessing = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:
        """
        The standard imagenet transforms for validation
        """

        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                imagenet_normalization(),
            ]
        )
        return preprocessing


class ImageNetNoAugDataModule(ImageNetDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        num_workers=16,
        image_size=224,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=224,
        )

    def train_transform(self) -> Callable:
        """
        Override to use inference transforms
        """
        return self.val_transform()


class ImageNetTopAugmentationModule(ImageNetDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        num_workers: int = 16,
        top_transforms_dir: str = None,
        num_transforms: int = 25,
        similarity_type: str = "resnet18_no_aug",
        plus_standard_aug: bool = False,
        standard_aug_before: bool = True,
        top_per_class: bool = True,
        top_transform_ranking: str = "proportion_boosted",
        transform_prob: dict = {"dist": "weighted_boost"},
        min_prop_boosted_filter: Optional[float] = 0.4,
        min_perc_change_per_class_filter: Optional[float] = 0.0,
    ):
        super().__init__(
            data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
        )
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = 224
        self.num_workers = num_workers
        self.top_transforms_dir = top_transforms_dir
        self.num_transforms = num_transforms
        self.similarity_type = similarity_type
        self.plus_standard_aug = plus_standard_aug
        self.standard_aug_before = standard_aug_before
        self.top_per_class = top_per_class
        self.top_transform_ranking = top_transform_ranking
        self.transform_prob = transform_prob
        self.min_prop_boosted_filter = min_prop_boosted_filter
        self.min_perc_change_per_class_filter = min_perc_change_per_class_filter

    def _create_dataloader(self, stage: str, augmentations: transform_lib.Compose):
        path = os.path.join(self.data_dir, stage)

        if stage == "train":
            shuffle = True
            dataset = TopAugmentationsDataset(
                path,
                transform_dir=self.top_transforms_dir,
                num_transforms=self.num_transforms,
                similarity_type=self.similarity_type,
                plus_standard_aug=self.plus_standard_aug,
                standard_aug_before=self.standard_aug_before,
                top_per_class=self.top_per_class,
                top_transform_ranking=self.top_transform_ranking,
                transform_prob=self.transform_prob,
                min_prop_boosted_filter=self.min_prop_boosted_filter,
                min_perc_change_per_class_filter=self.min_perc_change_per_class_filter,
            )
        else:
            shuffle = False
            dataset = torchvision.datasets.ImageFolder(path, augmentations)

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
        return data_loader


class ImageNetSingleClassAugmentationModule(ImageNetDataModule):
    """Applies a single transform to the specified class during training"""

    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        num_workers: int = 16,
        class_label: str = "n03291819",
        transform: str = "shearX",
        magnitude: int = 5,
    ):
        super().__init__(
            data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
        )
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = 224
        self.num_workers = num_workers

        self.class_label = class_label
        self.transform_name = transform
        self.magnitude = magnitude

    def create_transform(self):
        return augmentations_lib.Augmentation(
            self.transform_name, self.magnitude, uniformly_sample_magnitude=True
        )

    def _create_dataloader(self, stage: str, augmentations: transform_lib.Compose):
        path = os.path.join(self.data_dir, stage)

        if stage == "train":
            shuffle = True
            class_label_to_transform = {self.class_label: [self.create_transform()]}
            dataset = ManualAugmentationDataset(path, class_label_to_transform)
        else:
            shuffle = False
            dataset = torchvision.datasets.ImageFolder(path, augmentations)

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
        return data_loader
