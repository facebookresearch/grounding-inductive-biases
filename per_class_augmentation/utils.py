"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import pandas as pd
from per_class_augmentation import data
import numpy as np


def get_checkpoint_dir(subdir="per_class_augmentation/2021-04-03_09-43-44/") -> str:
    logs_dir = Path("")
    checkpoint_dir = str(list((logs_dir / subdir).rglob("*.ckpt"))[0])
    return checkpoint_dir


def plot_normalized_image(image):
    to_pil = torchvision.transforms.ToPILImage()
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )
    pil_image = to_pil(inv_normalize(image).squeeze())
    plt.imshow(pil_image)


def load_class_labels() -> List[str]:
    class_labels = pd.read_csv(
        "",
        sep=" ",
        names=["class_label", "idx", "name"],
    )["class_label"].values.tolist()
    return class_labels


def count_remaining_transforms(
    min_prop_boosted: Optional[float] = None,
    min_perc_change: Optional[float] = 0.4,
    data_dir: str = "",
    top_transforms_dir: str = "",
    top_transform_ranking: str = "avg_percent_similarity_change",
    num_transforms: int = 25,
):
    top_augs = data.TopAugmentationsDataset(
        data_dir + "/train",
        min_prop_boosted,
        num_transforms=num_transforms,
        transform_dir=top_transforms_dir,
        similarity_type="resnet18",
        plus_standard_aug=True,
        top_per_class=False,
        top_transform_ranking=top_transform_ranking,
        min_perc_change_per_class_filter=min_perc_change,
    )
    top_count = len(top_augs.top_augmentations.get_top_transforms("n1"))
    print("top across ", top_count, " out of ", num_transforms)
    top_augs_per_class = data.TopAugmentationsDataset(
        data_dir + "/train",
        min_prop_boosted,
        num_transforms=num_transforms,
        transform_dir=top_transforms_dir,
        similarity_type="resnet18",
        plus_standard_aug=True,
        top_per_class=True,
        top_transform_ranking=top_transform_ranking,
        min_perc_change_per_class_filter=min_perc_change,
    )

    counts = []
    class_labels = load_class_labels()
    for class_label in class_labels:
        count = len(
            top_augs_per_class.top_augmentations.get_top_transforms(class_label)
        )
        counts.append(count)
    top_count_per_class = np.average(counts)
    print("top per class ", top_count_per_class, " out of ", num_transforms)