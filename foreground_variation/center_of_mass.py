"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from scipy.ndimage.measurements import center_of_mass as center_of_mass_scipy
from collections import defaultdict
from typing import List


def measure_centers_of_mass(
    data_loader: DataLoader, idx_to_class: dict, max_samples=None
):
    """Measure center of masses per class.

    Args:
        data_loader: contains images and class index
        idx_to_class: maps class index to class label
        max_samples: limits number of samples to use.
            If None, iterates over entire dataset.
    """
    label_to_centers = defaultdict(list)

    for i, (x, class_idx) in enumerate(data_loader):
        center = center_of_mass(x)
        label = idx_to_class[int(class_idx)]
        label_to_centers[label].append(center)
        if max_samples is not None and i >= max_samples:
            break
    return label_to_centers


def center_of_mass(x: torch.Tensor) -> tuple:
    if x.squeeze().dim() != 2:
        raise ValueError(f"input must be a single 2D image, not {x.shape=}")
    x_np = x.squeeze().numpy()
    return center_of_mass_scipy(x_np)


def save_centers(label_to_centers: dict, results_dir: str):
    """Saves centers per class as json"""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "centers.json")
    with open(path, "w") as f:
        json.dump(label_to_centers, f)


def save_variations(variations: List, results_dir: str):
    """Saves centers per class as json"""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "variations.csv")
    df = pd.DataFrame.from_records(variations)
    df.to_csv(path)


def compute_variations(label_to_centers: dict) -> List:
    """Returns a records dict in a format for pandas"""
    variations = []

    for label, centers in label_to_centers.items():
        centers_x = np.array([c[0] for c in centers])
        centers_y = np.array([c[1] for c in centers])
        std_x, std_y = np.nanstd(centers_x), np.nanstd(centers_y)
        variations.append({"label": label, "std_x": std_x, "std_y": std_y})
    return variations


def main(data_module: pl.LightningDataModule, max_samples=None, results_dir=None):
    idx_to_class = data_module.idx_to_class
    label_to_centers = measure_centers_of_mass(
        data_module.train_dataloader(),
        idx_to_class,
        max_samples=max_samples,
    )
    # compute spread
    variations = compute_variations(label_to_centers)
    # save results
    if results_dir:
        save_centers(label_to_centers, results_dir)
        save_variations(variations, results_dir)
