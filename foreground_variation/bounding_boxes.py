"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Computes bounding boxes for foregrounds
"""

import torch
import json
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from typing import Tuple
from collections import defaultdict
from torch.utils.data import DataLoader


def find_bounding_box(img: torch.Tensor, threshold=0.1) -> Tuple[np.array, np.array]:
    """Returns top left and bottom right coordinates of box"""
    indices_above_threshold = (img.squeeze() > threshold).nonzero()

    top_left_indices = indices_above_threshold.min(dim=0).values.numpy()
    bottom_right_indices = indices_above_threshold.max(dim=0).values.numpy()
    return top_left_indices, bottom_right_indices


def find_bounding_boxes(
    data_loader: DataLoader, idx_to_class: dict, threshold=0.1, max_samples=None
) -> dict[str, list]:
    """Computes top_left and bottom_right bounding box coordinates.

    Args:
        data_loader: contains images and class index
        idx_to_class: maps class index to class label
        treshold (float): threshold to use for computing edge of box.
        max_samples: limits number of samples to use.
            If None, iterates over entire dataset.

    Returns: np.array of top_left and bottom_right indices for bounding boxes
    """
    label_to_top_left = defaultdict(list)
    label_to_bottom_right = defaultdict(list)

    if data_loader.batch_size != 1:
        raise ValueError("batch size must be 1.")

    num_skipped = 0

    for i, (image, class_idx) in enumerate(data_loader):
        try:
            top_left, bottom_right = find_bounding_box(image, threshold=threshold)
        except RuntimeError:
            num_skipped += 1
            continue
        label = idx_to_class[int(class_idx)]
        label_to_top_left[label].append(top_left)
        label_to_bottom_right[label].append(bottom_right)
        if i % 1000 == 0:
            print("i", i)
    print("total_processed", i)
    print("skipped", num_skipped)
    return label_to_top_left, label_to_bottom_right


def compute_areas(top_left: np.array, bottom_right: np.array) -> np.array:
    sizes = np.abs(top_left - bottom_right)
    return sizes[:, 0] * sizes[:, 1]


def build_results_df(
    label_to_top_left: dict, label_to_bottom_right: dict
) -> pd.DataFrame:
    records = []
    for label in label_to_top_left:
        top_left = np.array(label_to_top_left[label])
        bottom_right = np.array(label_to_bottom_right[label])
        areas = compute_areas(top_left, bottom_right)
        record = {
            "top_left_x_50%": np.percentile(top_left[0], 50),
            "top_left_x_25%": np.percentile(top_left[0], 25),
            "top_left_x_75%": np.percentile(top_left[0], 75),
            "top_left_y_50%": np.percentile(top_left[1], 50),
            "top_left_y_25%": np.percentile(top_left[1], 25),
            "top_left_y_75%": np.percentile(top_left[1], 75),
            "bottom_right_x_50%": np.percentile(bottom_right[0], 50),
            "bottom_right_x_25%": np.percentile(bottom_right[0], 25),
            "bottom_right_x_75%": np.percentile(bottom_right[0], 75),
            "bottom_right_y_50%": np.percentile(bottom_right[1], 50),
            "bottom_right_y_25%": np.percentile(bottom_right[1], 25),
            "bottom_right_y_75%": np.percentile(bottom_right[1], 75),
            "areas_50%": np.percentile(areas, 50),
            "areas_25%": np.percentile(areas, 25),
            "areas_75%": np.percentile(areas, 75),
            "label": label,
        }
        records.append(record)
    df = pd.DataFrame.from_records(records)
    return df


def save_coordinates(label_to_coordinate: dict, results_dir: str, name="top_left"):
    """Saves centers per class as json"""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    label_to_coordinate_list = convert_numpy_to_list(label_to_coordinate)
    with open(path, "w") as f:
        json.dump(label_to_coordinate_list, f)


def convert_numpy_to_list(label_to_coordinate: dict) -> dict[str, list]:
    """Converts nested numpy dict values to python lists for saving"""
    label_to_coordinate_list = dict()

    for label in label_to_coordinate:
        label_to_coordinate_list[label] = np.array(label_to_coordinate[label]).tolist()
    return label_to_coordinate_list


def save_df(df: pd.DataFrame, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "bounding_boxes.csv")
    df.to_csv(path)


def main(data_module: pl.LightningDataModule, max_samples=None, results_dir=None):
    idx_to_class = data_module.idx_to_class
    label_to_top_left, label_to_bottom_right = find_bounding_boxes(
        data_module.train_dataloader(),
        idx_to_class,
        max_samples=max_samples,
    )

    df = build_results_df(label_to_top_left, label_to_bottom_right)

    # save results
    if results_dir:
        save_df(df, results_dir)
        save_coordinates(label_to_top_left, results_dir, name="top_left")
        save_coordinates(label_to_bottom_right, results_dir, name="bottom_right")