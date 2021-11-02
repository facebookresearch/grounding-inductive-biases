"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from image_similarity.image_embedding import Embedding
import torch
from image_similarity import similarity


def test_embedding_similarity():
    image_batch1 = torch.rand([10, 3, 224, 224])
    image_batch2 = torch.rand([10, 3, 224, 224])

    similarities = similarity.embedding_similarity(image_batch1, image_batch2)
    assert similarities.shape == (10,)


def test_embedding_similarity_single_image_batch():
    image_1 = torch.rand([1, 3, 224, 224])
    image_2 = torch.rand([1, 3, 224, 224])

    similarities = similarity.embedding_similarity(image_1, image_2)
    assert similarities.shape == (1,)
    assert type(similarities.item()) is float


def test_embedding_similarity_single_image():
    image_1 = torch.rand([3, 224, 224])
    image_2 = torch.rand([3, 224, 224])

    similarities = similarity.embedding_similarity(image_1, image_2)
    assert similarities.shape == (1,)
    assert type(similarities.item()) is float


def test_get_embedding_model():
    embedding = similarity.get_embedding_model("resnet18")
    assert isinstance(embedding, Embedding)
    embedding_no_aug = similarity.get_embedding_model("resnet18_no_aug")
    assert isinstance(embedding_no_aug, Embedding)


def test_embedding_resnet_no_aug():
    image_batch1 = torch.rand([10, 3, 224, 224])
    image_batch2 = torch.rand([10, 3, 224, 224])

    similarities = similarity.embedding_similarity(
        image_batch1, image_batch2, model_type="resnet18_no_aug"
    )
    assert similarities.shape == (10,)