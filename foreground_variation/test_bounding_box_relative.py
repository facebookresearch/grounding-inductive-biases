"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pytest
import torch
import foreground_variation
from foreground_variation import bounding_boxes_relative
from foreground_variation import data
from foreground_variation.bounding_boxes_relative import compute_center, compute_area


class TestRelativeBoundingBox:
    def test_compute_center(self):
        center_x, center_y = compute_center(np.array([0.0, 5.0]), np.array([1.0, 1.0]))
        assert center_x == 0.5
        assert center_y == 3.0

    def test_compute_areas(self):
        area = compute_area(np.array([0.0, 5.0]), np.array([1.0, 1.0]))
        assert area == 4.0

    def test_find_bounding_box(self):
        image_batch = torch.rand(1, 1, 224, 224)
        top_left, bottom_right = bounding_boxes_relative.find_bounding_box(image_batch)
        assert top_left.shape == (2,)
        assert bottom_right.shape == (2,)

    @pytest.mark.slow
    def test_data_module(self):
        data_module = data.ImageNetForegroundModule()
        (image, _) = next(iter(data_module.train_dataloader()))
        assert image.shape[0] == 1
        assert image.shape[1] == 1

    def test_results_update(self):
        results = bounding_boxes_relative.Results()
        results.update("n1", 0.5, 1.5, 4.0)
        assert results.center_x.percentile(50) == 0.5

    def test_results_save_load(self, tmp_path):
        results = bounding_boxes_relative.Results()
        results.update("n1", 0.5, 1.5, 4.0)
        results.save(tmp_path)
        new_results = bounding_boxes_relative.Results()
        new_results.load(tmp_path)
        assert new_results.center_x.percentile(50) == 0.5
