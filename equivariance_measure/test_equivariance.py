import pytest
import torch
import numpy as np
from tdigest import TDigest
import pytorch_lightning as pl
from equivariance_measure.embedding_alignments import (
    TransformAlignmentsDigest,
    EmbeddingAlignmentModule,
    AlignmentsDigest,
)


@pytest.fixture(scope="session")
def dataloader():
    dataset = torch.utils.data.TensorDataset(
        torch.rand(4, 3, 224, 224), torch.ones(4, 1)
    )
    my_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=6)
    return my_dataloader


@pytest.fixture(scope="session")
def model(dataloader):
    model = EmbeddingAlignmentModule()
    trainer = pl.Trainer(gpus=1)
    trainer.test(
        model=model,
        test_dataloaders=dataloader,
    )
    return model


def test_embedding_module(model):
    assert model.magnitude_to_diff[3].shape == (4, 512)
    assert len(model.alignments) == 10
    assert model.alignments[3].shape == (4,)
    digest = model.alignments_digest.magnitude_to_digest[3]
    assert isinstance(digest, TDigest)


def test_shuffle_d_matrix():
    model = EmbeddingAlignmentModule()
    d_matrix = torch.rand(3, 512)
    d_shuffled = model.shuffle_d_matrix(d_matrix)
    assert d_shuffled.shape == (3, 512)
    assert (d_shuffled > 0).all()


def test_alignments_digest(model):
    alignments = TransformAlignmentsDigest("t1", data_stage="val")
    alignments.batch_update({"0": torch.rand(10).numpy(), "1": torch.rand(10).numpy()})
    alignments_dict = alignments.serialize()
    assert "1" in alignments_dict
    assert isinstance(alignments_dict["1"], dict)


class TestMultiTransformAlignments:
    def test_update(self, model):
        alignments = AlignmentsDigest("val")
        alignments.update(model)
        assert isinstance(
            alignments.transform_to_alignments_digest["shearX"],
            TransformAlignmentsDigest,
        )

    def test_save(self, model, tmp_path):
        alignments = AlignmentsDigest("val")
        alignments.update(model)
        alignments.save(tmp_path)
        assert (tmp_path / "alignments_val.json").is_file()

    def test_load(self, model, tmp_path):
        alignments = AlignmentsDigest("val")
        alignments.update(model)
        alignments.save(tmp_path)

        new_alignments = AlignmentsDigest("val")
        new_alignments.load(tmp_path)
        assert isinstance(
            new_alignments.transform_to_alignments_digest["shearX"],
            TransformAlignmentsDigest,
        )

    def test_get_percentile(self, model):
        alignments = AlignmentsDigest("val")
        alignments.update(model)
        assert alignments.get_percentile("shearX", 3, 50) > 0
