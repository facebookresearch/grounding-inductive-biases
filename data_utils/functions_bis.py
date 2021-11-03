"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.folder import DatasetFolder, default_loader
from utils.mytransforms import RandomSizeResizedCenterCrop
import numpy as np

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class MyImageFolder(DatasetFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(MyImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

def return_augmentations_types(args):

    if args.tvalues is None:
        args.tvalues = [1,1]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    scale_magnitudes = np.linspace(1,6,15)
    interpolation = transforms.RandomResizedCrop(224).interpolation
    augmentations = {}
    print(args.scale_mag, 1./scale_magnitudes[args.scale_mag]**2)
    interpolation = transforms.RandomResizedCrop(224).interpolation
    augmentations["C"] = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            224
                        ),  # resize is not needed since the crop will be in function of the size and ratio of the image
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
    augmentations["Cvary"] = transforms.Compose(
            [
                transforms.Resize(256),  # ensures that the minimal size is 256
                transforms.CenterCrop(224),
                RandomSizeResizedCenterCrop(
                    224, 
                    scale=(1./scale_magnitudes[args.scale_mag]**2,1.),
                    ratio=(1.,1.), interpolation=interpolation
                ),  
                transforms.ToTensor(),
                normalize,
            ]
        )
    augmentations["None"] = transforms.Compose(
            [
                transforms.Resize(256),  # ensures that the minimal size is 256
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    
    return augmentations 

def return_loader_and_sampler(args, traindir, valdir, return_train = True):

    augmentations = return_augmentations_types(args)

    if return_train:
        train_dataset = MyImageFolder(
                traindir, augmentations[args.augment_train])
    else:
        train_dataset = []

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # per GPU for DistributedDataParallel
        batch_size = int(args.batch_size / args.world_size)
        print(f"batch size per GPU is {batch_size}")
    else:
        train_sampler = None
        batch_size = args.batch_size

    if return_train:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
    else:
        train_loader = None
    print("Train loader initiated")
    val_loader = torch.utils.data.DataLoader(
        MyImageFolder(
            valdir,
            augmentations[args.augment_valid]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    print("Val loader initiated")
    return train_loader, val_loader, train_sampler