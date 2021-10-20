"""
Tests foreground analysis
"""
import torch
import numpy as np
from foreground_variation.data import ImageNetForegroundModule
from foreground_variation import center_of_mass, bounding_boxes


def test_foreground_data():
    batch_size = 1
    data_module = ImageNetForegroundModule(batch_size=batch_size)
    data_loader = data_module.train_dataloader()
    assert data_module.idx_to_class[17] == "n01580077"
    assert data_loader.batch_size == batch_size

    image, label = next(iter(data_loader))
    assert image.shape == torch.Size((1, 1, 224, 224))
    assert label.shape == torch.Size((1,))


def test_center_of_mass():
    x = torch.rand(1, 224, 224)
    center = center_of_mass.center_of_mass(x)
    assert len(center) == 2
    assert center[0] > 0.0
    assert center[1] > 0.0


def test_measure_centers_of_mass():
    class_idx = torch.tensor([0, 0, 1, 0, 1])
    dataset = torch.utils.data.TensorDataset(torch.rand(5, 1, 224, 224), class_idx)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    idx_to_class = {0: "apple", 1: "banana"}
    label_to_center = center_of_mass.measure_centers_of_mass(data_loader, idx_to_class)
    assert len(label_to_center["apple"]) == 3
    assert len(label_to_center["banana"]) == 2
    # a single element should be a tuple
    assert len(label_to_center["apple"][2]) == 2


def test_center_of_mass_main():
    data_module = ImageNetForegroundModule(batch_size=1)
    center_of_mass.main(data_module, max_samples=10)


def test_find_bounding_boxes():
    class_idx = torch.tensor([0, 0, 1, 0, 1])
    dataset = torch.utils.data.TensorDataset(torch.rand(5, 1, 224, 224), class_idx)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    idx_to_class = {0: "apple", 1: "banana"}
    top_left, bottom_right = bounding_boxes.find_bounding_boxes(
        data_loader, idx_to_class
    )
    assert "apple" in top_left
    assert len(top_left) == 2
    assert len(top_left["apple"]) == 3


def test_build_results_df():
    label_to_top_left = {
        "n1": [np.array([0, 1]), np.array([1, 2])],
        "n2": [np.array([0, 1]), np.array([1, 2])],
    }
    label_to_bottom_right = {
        "n1": [np.array([200, 150]), np.array([230, 220])],
        "n2": [np.array([201, 151]), np.array([231, 221])],
    }
    df = bounding_boxes.build_results_df(label_to_top_left, label_to_bottom_right)
    assert len(df) == 2


