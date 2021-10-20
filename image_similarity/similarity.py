from image_similarity.image_embedding import Embedding
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from functools import lru_cache
import torch
import torchvision
import numpy as np


def embedding_similarity(
    image_batch_1: torch.Tensor,
    image_batch_2: torch.Tensor,
    model_type: str = "resnet18",
) -> np.ndarray:
    """
    Computes cos similarity of embeddings

    Args:
        model_type (str): model type used for embedding. Options are "resnet18" or "resnet18_no_aug"
    """
    embedding = get_embedding_model(model_type=model_type)
    image_batch_1, image_batch_2 = image_batch_1.squeeze(), image_batch_2.squeeze()
    if image_batch_1.dim() == 3:
        return embedding_similarity_single_images(
            image_batch_1, image_batch_2, model_type=model_type
        )
    image_1_vecs = embedding.embed(image_batch_1)
    image_2_vecs = embedding.embed(image_batch_2)

    similarities = []

    for image_1_vec, image_2_vec in zip(image_1_vecs, image_2_vecs):
        similarity = cosine_similarity(
            image_1_vec.unsqueeze(0), image_2_vec.unsqueeze(0)
        ).item()

        similarities.append(similarity)

    return np.array(similarities)


# ensures ResNet is loaded only once on import
@lru_cache
def get_embedding_model(model_type: str):
    return Embedding(model_name=model_type, cuda=torch.cuda.is_available())


def embedding_similarity_single_images(image_1, image_2, model_type="resnet18"):
    embedding = get_embedding_model(model_type)
    image_1_vec = embedding.embed(image_1.unsqueeze(0))
    image_2_vec = embedding.embed(image_2.unsqueeze(0))
    similarity = cosine_similarity(
        image_1_vec.unsqueeze(0), image_2_vec.unsqueeze(0)
    ).item()
    return np.array([similarity])


def to_image(image_tensor: torch.Tensor):
    """Converts input into a PIL image"""
    to_pil = torchvision.transforms.ToPILImage()
    return to_pil(image_tensor.squeeze())


def to_batch(image):
    """Converts a single image to a batch"""
    if image.dim() == 4:
        return image
    return image.unsqueeze(0)
