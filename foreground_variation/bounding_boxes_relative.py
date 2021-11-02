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
from tdigest import TDigest
import numpy as np
import pytorch_lightning as pl
import plotly.graph_objects as go
from typing import Tuple, Dict
from collections import defaultdict
from torch.utils.data import DataLoader


class Results:
    def __init__(self):
        self.label_to_center_x = defaultdict(TDigest)
        self.label_to_center_y = defaultdict(TDigest)
        self.label_to_area = defaultdict(TDigest)

        self.center_x = TDigest()
        self.center_y = TDigest()
        self.area = TDigest()

        self.FILE_NAME_TO_DIGEST = {
            "center_x.json": self.center_x,
            "center_y.json": self.center_y,
            "area.json": self.area,
        }

        self.FILE_NAME_TO_LABEL_DIGEST = {
            "per_class_center_x.json": self.label_to_center_x,
            "per_class_center_y.json": self.label_to_center_y,
            "per_class_area.json": self.label_to_area,
        }

    def update(self, class_label: str, center_x: float, center_y: float, area: float):
        self.center_x.update(center_x)
        self.center_y.update(center_y)
        self.area.update(area)

        self.label_to_center_x[class_label].update(center_x)
        self.label_to_center_y[class_label].update(center_y)
        self.label_to_area[class_label].update(area)

    def _save_aggregate(self, results_dir: str):
        for file_name, digest in self.FILE_NAME_TO_DIGEST.items():
            path = os.path.join(results_dir, file_name)
            with open(path, "w") as f:
                json.dump(digest.to_dict(), f)

    def _save_label_to_digest(self, results_dir: str):
        for file_name, label_to_digest in self.FILE_NAME_TO_LABEL_DIGEST.items():
            path = os.path.join(results_dir, file_name)
            with open(path, "w") as f:
                serialized = self.serialize_label_digest(label_to_digest)
                json.dump(serialized, f)

    @staticmethod
    def serialize_label_digest(label_to_digest: Dict[str, TDigest]) -> dict:
        serialized = dict()
        for label, digest in label_to_digest.items():
            serialized[label] = digest.to_dict()
        return serialized

    def save(self, results_dir: str):
        os.makedirs(results_dir, exist_ok=True)
        self._save_aggregate(results_dir)
        self._save_label_to_digest(results_dir)

    def load(self, results_dir: str):
        for file_name, digest in self.FILE_NAME_TO_DIGEST.items():
            path = os.path.join(results_dir, file_name)
            with open(path) as f:
                digest_dict = json.load(f)
            digest.update_from_dict(digest_dict)

        self._load_label_digests(results_dir)

    def _load_label_digests(self, results_dir: str):
        for file_name, label_to_digest in self.FILE_NAME_TO_LABEL_DIGEST.items():
            path = os.path.join(results_dir, file_name)
            with open(path) as f:
                label_to_digest_dict = json.load(f)
            for label, digest_dict in label_to_digest_dict.items():
                label_to_digest[label].update_from_dict(digest_dict)

    def plot(self) -> go.Figure:
        """Plots distributions of aggregate values"""
        marker_size = 0.01
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                x=[self.area.percentile(i) for i in range(101)],
                name="area",
                marker_size=marker_size,
            )
        )
        fig.add_trace(
            go.Box(
                x=[self.center_x.percentile(i) for i in range(101)],
                name="center x",
                marker_size=marker_size,
            )
        )
        fig.add_trace(
            go.Box(
                x=[self.center_y.percentile(i) for i in range(101)],
                name="center y",
                marker_size=marker_size,
            )
        )
        fig.update_xaxes(title="relative position within image")
        return fig


def compute_results(
    data_loader: DataLoader, idx_to_class: dict, threshold=0.1, max_samples=None
) -> Results:
    """Computes center and area of bouding box relative to frame.

    Args:
        data_loader: contains images and class index
        idx_to_class: maps class index to class label
        treshold (float): threshold to use for computing edge of box.
        max_samples: limits number of samples to use.
            If None, iterates over entire dataset.

    Returns: Results object
    """

    if data_loader.batch_size != 1:
        raise ValueError("batch size must be 1.")

    results = Results()

    num_skipped = 0

    for i, (image, class_idx) in enumerate(data_loader):
        try:
            top_left, bottom_right = find_bounding_box(image, threshold=threshold)
        except RuntimeError:
            num_skipped += 1
            continue
        class_label = idx_to_class[int(class_idx)]
        width, height = float(image.shape[2]), float(image.shape[3])
        center_x, center_y = compute_center(top_left, bottom_right)
        area = compute_area(top_left, bottom_right)
        results.update(
            class_label, center_x / width, center_y / height, area / (width * height)
        )
        if i % 1000 == 0:
            print("i", i)
        if max_samples and i > max_samples:
            break
    print("total_processed", i)
    print("skipped", num_skipped)
    return results


def find_bounding_box(img: torch.Tensor, threshold=0.1) -> Tuple[np.array, np.array]:
    """Returns top left and bottom right coordinates of box"""
    indices_above_threshold = (img.squeeze() > threshold).nonzero()

    top_left_indices = indices_above_threshold.min(dim=0).values.numpy()
    bottom_right_indices = indices_above_threshold.max(dim=0).values.numpy()
    return top_left_indices, bottom_right_indices


def compute_center(
    top_left: np.array, bottom_right: np.array
) -> Tuple[np.array, np.array]:
    assert len(top_left.shape) <= 1, "top_left should be 1-d input"
    center_x = top_left[0] + (bottom_right[0] - top_left[0]) / 2.0
    center_y = bottom_right[1] + (top_left[1] - bottom_right[1]) / 2.0
    return center_x, center_y


def compute_area(top_left: np.array, bottom_right: np.array) -> np.array:
    sizes = np.abs(top_left - bottom_right)
    return sizes[0] * sizes[1]


def main(data_module: pl.LightningDataModule, max_samples=None, results_dir=None):
    idx_to_class = data_module.idx_to_class
    results = compute_results(
        data_module.train_dataloader(),
        idx_to_class,
        max_samples=max_samples,
    )

    if results_dir:
        results.save(results_dir)
    return results
