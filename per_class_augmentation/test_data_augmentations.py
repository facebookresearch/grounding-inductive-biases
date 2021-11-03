"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import pytest
import numpy as np
from per_class_augmentation.data import TopAugmentationsDataset
from per_class_augmentation.data import ImageNetTopAugmentationModule
from per_class_augmentation import augmentations


DATA_DIR = "/datasets01/imagenet_full_size/061417"


@pytest.fixture(scope="module")
def top_transforms_across_classes():
    return augmentations.TopWeightedAugmentations(
        num_transforms=3,
        similarity_type="resnet18",
        transform_prob=dict({"dist": "fixed", "fixed_prob": 0.2}),
        top_per_class=False,
    )


@pytest.fixture(scope="module")
def top_transforms_per_class():
    return augmentations.TopWeightedAugmentations(
        num_transforms=3,
        similarity_type="resnet18",
        transform_prob=dict({"dist": "fixed", "fixed_prob": 0.2}),
        top_per_class=True,
    )


def test_per_class_augmentation_dataset():
    transform_path = (
        ""
    )
    dataset = TopAugmentationsDataset(
        DATA_DIR + "/train",
        0.4,
        transform_dir=transform_path,
    )
    image, label = dataset.__getitem__(3)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)


def test_per_class_augmentation_data_module():
    data_module = ImageNetTopAugmentationModule(DATA_DIR + "/train", batch_size=32)
    assert data_module.batch_size == 32


class TestTopWeightedAugmentations:
    def test_load_top_transform_df(self, top_transforms_across_classes):
        df = top_transforms_across_classes.df
        assert "transform_name" in df.columns

    def test_find_top_across_classes(self, top_transforms_across_classes):
        top_df = top_transforms_across_classes.group_top_across_classes()
        assert len(top_df) == top_transforms_across_classes.num_transforms
        assert "transform_name" in top_df.columns

    def test_find_top_per_class(self, top_transforms_per_class):
        top_df = top_transforms_per_class.group_top_per_class()
        assert len(top_df) == (top_transforms_per_class.num_transforms * 1000)
        assert "transform_name" in top_df.columns
        assert "class_label" in top_df.columns

    def test_get_top_transform_per_class(self, top_transforms_per_class):
        top_transforms = top_transforms_per_class.get_top_transforms(
            class_label="n01440764"
        )
        assert isinstance(top_transforms, augmentations.MultipleAugmentations)
        assert (np.array(top_transforms.probabilities) == 0.2).all()

    def test_get_top_transform_across_classes(self, top_transforms_across_classes):
        top_transforms = top_transforms_across_classes.get_top_transforms()
        assert isinstance(top_transforms, augmentations.MultipleAugmentations)
        assert (np.array(top_transforms.probabilities) == 0.2).all()
