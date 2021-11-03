"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import random
import pytest
import torch
import numpy as np
import pytorch_lightning as pl
from equivariance_measure import transformations
from equivariance_measure.embedding_distances import (
    EmbeddingDistanceModule,
    InvariancesDigest,
    TransformInvarianceDigest,
)


@pytest.fixture
def dataloader():
    dataset = torch.utils.data.TensorDataset(
        torch.rand(4, 3, 224, 224), torch.ones(4, 1)
    )
    my_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=6)
    return my_dataloader


@pytest.mark.parametrize("transform_name", transformations.TRANSFORMATION_NAMES)
def test_transformation(transform_name):
    image = torch.rand(3, 224, 224)
    magnitude_idx = random.randint(0, 9)
    transform = transformations.Transformation(transform_name, magnitude_idx)
    image_transformed = transform(image)
    assert image_transformed.shape == (3, 224, 224)


def test_embedding_module(dataloader):
    model = EmbeddingDistanceModule()
    assert isinstance(model, pl.LightningModule)

    trainer = pl.Trainer(gpus=1)
    trainer.test(
        model=model,
        test_dataloaders=dataloader,
    )
    assert isinstance(model.cos_distance_digest, TransformInvarianceDigest)
    assert isinstance(model.l2_digest, TransformInvarianceDigest)


class TestInvarianceDigest:
    def test_invariance_digest(self, dataloader):
        model = EmbeddingDistanceModule()
        trainer = pl.Trainer(gpus=1)
        trainer.test(
            model=model,
            test_dataloaders=dataloader,
        )

        cos_distance_invariance_digest = InvariancesDigest("cos_distance", "train")
        l2_invariance_digest = InvariancesDigest("l2", "train")

        cos_distance_invariance_digest.update(model)
        l2_invariance_digest.update(model)

        assert isinstance(
            cos_distance_invariance_digest.transform_to_invariance_digest["shearX"],
            TransformInvarianceDigest,
        )
        assert isinstance(
            l2_invariance_digest.transform_to_invariance_digest["shearX"],
            TransformInvarianceDigest,
        )

        assert cos_distance_invariance_digest.get_percentile("shearX", 3, 50) > -1.0
        assert l2_invariance_digest.get_percentile("shearX", 3, 49) > -1.0

    def test_saving_loading(self, tmp_path, dataloader):
        model = EmbeddingDistanceModule()
        trainer = pl.Trainer(gpus=1)
        trainer.test(
            model=model,
            test_dataloaders=dataloader,
        )
        cos_distance_invariance_digest = InvariancesDigest("cos_distance", "train")
        cos_distance_invariance_digest.update(model)
        cos_distance_invariance_digest.save(str(tmp_path))
        assert (tmp_path / "cos_distance_invariance_train.json").is_file()

        new_cos_distance_invariance_digest = InvariancesDigest("cos_distance", "train")
        new_cos_distance_invariance_digest.load(str(tmp_path))
        assert new_cos_distance_invariance_digest.get_percentile(
            "shearX", 3, 50
        ) == cos_distance_invariance_digest.get_percentile("shearX", 3, 50)


class TestTransformDigest:
    def test_updates(self):
        transform_digest = TransformInvarianceDigest("t1", distance_type="l2")

        baseline = np.array([1.0, 1.0, 1.0])
        transformed_distance = np.array([0.5, 1.0, 1.0])
        invariance = np.divide(baseline - transformed_distance, baseline)

        transform_digest.batch_update("1", baseline, transformed_distance, invariance)
        assert transform_digest.get_percentile(1, 0) == 0.0
        assert transform_digest.get_baseline_percentile(1, 50) == 1.0

    def test_serialization(self):
        transform_digest = TransformInvarianceDigest("t1", distance_type="l2")

        baseline = np.array([1.0, 1.0, 1.0])
        transformed_distance = np.array([0.5, 1.0, 1.0])
        invariance = np.divide(baseline - transformed_distance, baseline)

        transform_digest.batch_update("1", baseline, transformed_distance, invariance)
        serialized_transforms = transform_digest.serialize()
        assert isinstance(serialized_transforms, dict)
        assert isinstance(serialized_transforms["baseline"]["1"], dict)
        assert isinstance(serialized_transforms["invariance"]["1"], dict)
