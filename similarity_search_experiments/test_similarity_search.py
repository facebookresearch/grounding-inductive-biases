"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import pytest
import numpy as np
import torch
from similarity_search_experiments import similarity_search
from image_similarity import load_pairs
from torch.utils.data import TensorDataset, DataLoader
from auto_augment.best_policies import ImageNetPolicy


class TestTransformations:
    def test_transform_single_image(self):
        image_net_policies = ImageNetPolicy()
        image = torch.rand([3, 224, 224])
        transformations = image_net_policies.get_unique_single_transformations()
        transformed_image = transformations[3].apply_single(image)
        assert transformed_image.shape == torch.Size([3, 224, 224])

    def test_transform_image_batch(self):
        image_net_policies = ImageNetPolicy()
        images = torch.rand([10, 3, 224, 224])
        transformations = image_net_policies.get_unique_single_transformations()
        transformed_image = transformations[3].apply(images)
        assert transformed_image.shape == torch.Size([10, 3, 224, 224])

    def test_subpolicy_image_batch(self):
        image_net_policies = ImageNetPolicy()
        images = torch.rand([10, 3, 224, 224])
        subpolicies = image_net_policies.get_unique_subpolicies()
        transformed_image = subpolicies[3].apply(images)
        assert transformed_image.shape == torch.Size([10, 3, 224, 224])


class TestBoost:
    def test_instantiation(self):
        originals = [1.5, 2.1, 3.1]
        changes = [0.2, -0.1, 0.3]
        # with numpy array
        boost = similarity_search.Boost(
            "my_transform", np.array(changes), np.array(originals)
        )
        assert boost.proportion_boosted == 2 / 3.0

        # verify instantiating with a list raises an exception
        with pytest.raises(ValueError):
            boost = similarity_search.Boost("my_transform", changes, originals)

        # verify instantiating with wrong sizes raises an exception
        with pytest.raises(ValueError):
            boost = similarity_search.Boost(
                "my_transform", np.array(changes), np.array(originals[:-1])
            )

    def test_boosts_to_dataframe(self):
        original = [11.1, 10.1, 10.11, 10.1, 11.1, 10.1, 10.9]
        boost1 = similarity_search.Boost(
            "t1", np.array([1.1, -0.1, 0.11, 0.1, -1.1, 0.1, 0.9]), np.array(original)
        )
        boost2 = similarity_search.Boost(
            "t2", np.array([1.2, -0.2, 1.2, 0.2, -1.2, 0.2, 0.9]), np.array(original)
        )

        df = similarity_search.boosts_to_dataframe([boost1, boost2], "n1")
        assert len(df) == 2
        assert (df.proportion_boosted > 0).all()
        assert (df.proportion_boosted <= 1.0).all()


@pytest.fixture
def class_data_loader(request):
    batch_size = 4
    if request.param == "real":
        image_net_dir = load_pairs.get_image_net_dir()
        class_labels = load_pairs.load_class_labels(image_net_dir)
        class_labels_subset = class_labels[:2]
        data_loaders = similarity_search.get_data_loaders(
            class_labels_subset, image_net_dir, batch_size
        )
        data_loader = data_loaders[0]
    else:
        dataset = TensorDataset(
            torch.rand([batch_size * 15, 3, 224, 224]),
            torch.ones([batch_size * 15, 1]),
        )
        data_loader = DataLoader(dataset, drop_last=True, batch_size=batch_size)
    return data_loader


@pytest.fixture
def transformations(request):
    image_net_policies = ImageNetPolicy()
    if request.param == "subpolicy":
        return image_net_policies.get_unique_subpolicies()[:3]
    return image_net_policies.get_unique_single_transformations()[:3]


class TestSimilaritySearch:
    @pytest.fixture
    def images(self):
        return torch.rand([10, 3, 224, 224])

    @pytest.fixture
    def class_data_loaders(self):
        data_loaders = []
        for i in range(2):
            dataset = TensorDataset(torch.rand([60, 3, 224, 224]), torch.ones([60, 1]))
            data_loader = DataLoader(dataset, drop_last=True, batch_size=4)
            data_loaders.append(data_loader)
        return data_loaders

    def test_select_partition(self):
        fake_labels = [f"a{i}" for i in range(100)]
        last_partition = similarity_search.select_partition(fake_labels, 10, 9)
        assert len(last_partition) == 10

        first_partition = similarity_search.select_partition(fake_labels, 5, 0)
        assert len(first_partition) == 20

    def test_get_data_loaders(self):
        image_net_dir = load_pairs.get_image_net_dir()
        class_labels = load_pairs.load_class_labels(image_net_dir)
        class_labels_subset = class_labels[:5]
        batch_size = 4

        data_loaders = similarity_search.get_data_loaders(
            class_labels_subset, image_net_dir, batch_size
        )

        assert len(data_loaders) == 5
        assert data_loaders[4].batch_size == batch_size

    @pytest.mark.parametrize("class_data_loader", ["real", "fake"], indirect=True)
    @pytest.mark.parametrize("transformations", ["single", "subpolicy"], indirect=True)
    @pytest.mark.parametrize("similarity_type", ["resnet18", "resnet18_no_aug"])
    def test_compute_class_similarity_boosts(
        self, class_data_loader, transformations, similarity_type
    ):
        max_n_image_pairs = 12
        boosts = similarity_search.compute_class_similarity_boosts(
            class_data_loader,
            transformations,
            max_n_image_pairs,
            similarity_type,
        )
        assert type(class_data_loader) is DataLoader
        assert len(boosts) == len(transformations)
        assert type(boosts[0]) is similarity_search.Boost
        # should contain a change per image pair
        assert boosts[0].similarity_changes.shape == (max_n_image_pairs,)
        assert boosts[0].original_similarities.shape == (max_n_image_pairs,)

    @pytest.mark.parametrize("transformations", ["single", "subpolicy"], indirect=True)
    @pytest.mark.parametrize("similarity_type", ["resnet18", "resnet18_no_aug"])
    def test_compute_similarity_boosts(
        self, class_data_loaders, transformations, similarity_type
    ):
        max_n_image_pairs = 12
        labels = [f"n{i}" for i in range(len(class_data_loaders))]
        df = similarity_search.compute_similarity_boosts(
            class_data_loaders,
            labels,
            transformations,
            max_n_image_pairs,
            similarity_type,
        )
        expected_size = len(class_data_loaders) * len(transformations)
        assert len(df) == expected_size
        assert "class_label" in df.columns

        print("total_num_pairs", df.total_num_pairs)
        assert (df.total_num_pairs == max_n_image_pairs).all()
