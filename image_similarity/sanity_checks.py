"""
Sanity checks similarity metric using ImageNet validation images

1) across versus within class similarity: measure the difference in similarity between within- versus across- classes 
(hopefully, images of dogs are more similar to each other than those of cats) 
2) transformed_similarity: the extend to which similarity reflects transformation parameters (is 5° rotation versus 90° rotation more similar to the original image?). 
3) trasnformed_across_versus_within_class_similarity: comparing intra- (within) vs. inter-class similarity for a fixed set of transformation parameters.

Test Locally:
$ python sanity_checks.py hydra/launcher=submitit_local sanity_checks.max_n_image_pairs=5 sanity_checks.max_class_label_combinations=5 sanity_checks.run_on_sample_classes=True --multirun

--multi-run: is needed to work with submitit launcher

Run on cluster:
$ python sanity_checks.py --multirun
"""
import hydra
import logging
import warnings
import torchvision
from omegaconf import DictConfig
from functools import partial
from typing import Dict, List, Callable, Optional
from random import randrange
import torch
import submitit
import itertools
import numpy as np
from image_similarity.similarity import embedding_similarity
from pathlib import Path
from torch.utils.data import DataLoader
from image_similarity.load_pairs import (
    get_image_net_dir,
    get_class_data_loader,
    get_across_classes_pairs_data_loader,
    load_class_labels,
)

import pandas as pd

log = logging.getLogger(__name__)


def compute_within_class_similarity(
    pairs_loader: DataLoader,
    max_n_image_pairs: int,
    pairs_transform: Optional[Callable],
    similarity_type: str,
):
    """Computes within similarity for specified number of pairs n_pairs

    Args:
        pairs_loader: dataloader yielding pairs for a single class
        max_n_image_pairs: max number of image pairs to sample for similarity

    Returns: list of similarities for each pair
    """
    similarities = []

    for (image_1, image_2), (_, _) in pairs_loader:
        if pairs_transform:
            image_1, image_2 = pairs_transform(image_1, image_2)
        similarity = embedding_similarity(image_1, image_2, model_type=similarity_type)
        similarities.append(similarity.item())
        if len(similarities) > max_n_image_pairs:
            break
    return similarities


def compute_across_class_similarity(
    across_class_pairs_loader: DataLoader,
    max_n_image_pairs: int,
    pairs_transform: Optional[Callable],
    similarity_type: str,
):
    """Computes within similarity for specified number of pairs n_pairs

    Args:
        across_class_pairs_loader: dataloader yielding pairs for a single class
        max_n_image_pairs: max number of image pairs to sample for similarity

    Returns: list of similarities for each pair
    """
    similarities = []

    for (image_1, _), (image_2, _) in across_class_pairs_loader:
        if pairs_transform:
            image_1, image_2 = pairs_transform(image_1, image_2)
        similarity = embedding_similarity(image_1, image_2, model_type=similarity_type)
        similarities.append(similarity.item())
        if len(similarities) > max_n_image_pairs:
            break
    return similarities


def compute_within_class_similarities(
    image_net_dir: Path,
    max_n_image_pairs: int,
    class_labels: List[str],
    similarity_type: str,
    randomly_rotate: bool = False,
    pairs_transform: Optional[Callable] = None,
) -> Dict[str, List[float]]:
    """Pairwise similarity within each class in given labels"""
    class_to_similarities = {}
    for i, class_label in enumerate(class_labels):
        pairs_loader = get_class_data_loader(
            image_net_dir,
            batch_size=2,
            class_label=class_label,
            randomly_rotate=randomly_rotate,
        )
        within_class_similarity = compute_within_class_similarity(
            pairs_loader,
            max_n_image_pairs,
            pairs_transform,
            similarity_type,
        )
        class_to_similarities[class_label] = within_class_similarity
        if i % 100 == 0:
            log.debug(
                f"within similarity {i} classes out of {len(class_labels)} complete"
            )
    return class_to_similarities


def compute_across_class_similarities(
    image_net_dir: Path,
    max_n_image_pairs: int,
    class_labels: List[str],
    max_class_label_combinations: int,
    similarity_type: str,
    randomly_rotate: bool = False,
    pairs_transform: Optional[Callable] = None,
) -> Dict[str, List[float]]:
    """Pairwise similarity across classes for each class in given labels.

    Args:
        max_n_image_pairs: number of image pairs to sample
        max_class_label_combinations: max number of class labels pairs to sample
        similarity_type (str): model to use for embeddings in similarity metric. Example: "resnet18"
    """
    classes_to_similarities = {}
    for class_label_1, class_label_2 in itertools.islice(
        itertools.combinations(class_labels, 2), max_class_label_combinations
    ):
        across_class_pairs_loader = get_across_classes_pairs_data_loader(
            image_net_dir,
            class_label_1=class_label_1,
            class_label_2=class_label_2,
            randomly_rotate=randomly_rotate,
        )
        across_class_similarity = compute_across_class_similarity(
            across_class_pairs_loader,
            max_n_image_pairs,
            pairs_transform,
            similarity_type,
        )
        label = f"{class_label_1} {class_label_2}"
        classes_to_similarities[label] = across_class_similarity
    return classes_to_similarities


def compute_class_avg(class_similarities):
    return np.average(
        [np.average(similarities) for similarities in class_similarities.values()]
    )


def compute_class_std(class_similarities):
    return np.average(
        [np.std(similarities) for similarities in class_similarities.values()]
    )


def run_within_v_across(
    image_net_dir: Path,
    class_labels: List[str],
    similarity_type: str,
    max_n_image_pairs: int = 100,
    max_class_label_combinations: int = 100,
    randomly_rotate: bool = False,
    pairs_transform: Optional[Callable] = None,
) -> Dict[str, float]:
    """Compares within versus across class pairwise image similarity"""
    within_class_similarities = compute_within_class_similarities(
        image_net_dir,
        max_n_image_pairs,
        class_labels,
        similarity_type,
        randomly_rotate=randomly_rotate,
        pairs_transform=pairs_transform,
    )
    log.info(f"{'transformed' if randomly_rotate else ''} within class similarity done")
    across_class_similarities = compute_across_class_similarities(
        image_net_dir,
        max_n_image_pairs,
        class_labels,
        max_class_label_combinations,
        similarity_type,
        randomly_rotate=randomly_rotate,
        pairs_transform=pairs_transform,
    )
    log.info(f"{'transformed' if randomly_rotate else ''} across class similarity done")

    avg_within = compute_class_avg(within_class_similarities)
    std_within = compute_class_std(within_class_similarities)

    avg_across = compute_class_avg(across_class_similarities)
    std_across = compute_class_std(across_class_similarities)

    result = {
        f"avg {'transformed ' if randomly_rotate else ''}within": avg_within,
        f"std {'transformed ' if randomly_rotate else ''}within": std_within,
        f"avg {'transformed ' if randomly_rotate else ''}across": avg_across,
        f"std {'transformed ' if randomly_rotate else ''}across": std_across,
    }

    return result


def run_within_v_across_indep_transformed(
    image_net_dir: Path,
    class_labels: List[str],
    similarity_type: str,
    max_n_image_pairs: int = 100,
    max_class_label_combinations: int = 100,
) -> Dict[str, float]:
    """Compares within versus across class pairwise image similarity for rotated images.
    Each image is randomly rotated.
    """
    return run_within_v_across(
        image_net_dir,
        class_labels,
        similarity_type,
        max_n_image_pairs=max_n_image_pairs,
        max_class_label_combinations=max_class_label_combinations,
        randomly_rotate=True,
        pairs_transform=None,
    )


def run_within_v_across_pairwise_transformed(
    image_net_dir: Path,
    class_labels: List[str],
    similarity_type: str,
    max_n_image_pairs: int = 100,
    max_class_label_combinations: int = 100,
):
    """Compares within versus across class pairwise image similarity for rotated images.
    Each pair of images is identically rotated (with a random degree).
    """
    return run_within_v_across(
        image_net_dir,
        class_labels,
        similarity_type,
        max_n_image_pairs=max_n_image_pairs,
        max_class_label_combinations=max_class_label_combinations,
        randomly_rotate=False,
        pairs_transform=randomly_rotate_pair,
    )


def compute_transformed_similarity(
    class_loader: DataLoader,
    transformation: Callable,
    max_n_image_pairs: int,
    similarity_type: str,
):
    """Computes similarity between an image and its transformed version

    Args:
        class_loader: dataloader yielding images from a single class
        transformation: applies desired transformation
        similarity_type: model to use for embeddings in similarity metric. Example: "resnet18"

    Returns: list of similarities for each pair
    """
    similarities = []

    for image, _ in class_loader:
        transformed_image = transformation(image)
        similarity = embedding_similarity(
            image, transformed_image, model_type=similarity_type
        )
        similarities.append(similarity.item())
        if len(similarities) > max_n_image_pairs:
            break
    return similarities


def rotate(image: torch.Tensor, angle: float):
    rotated_image = torchvision.transforms.functional.rotate(image, angle)
    return rotated_image


def randomly_rotate_pair(image_1: torch.Tensor, image_2: torch.Tensor):
    """Applies the same random rotation to image 1 and image 2"""
    angle = float(randrange(360))
    image_1_rotated = rotate(image_1, angle)
    image_2_rotated = rotate(image_2, angle)
    return image_1_rotated, image_2_rotated


def compute_transformed_similarities(
    image_net_dir: Path,
    class_labels: List[str],
    transformation: Callable,
    max_n_image_pairs: int,
    similarity_type: str,
) -> Dict[str, List[float]]:
    """Computes transformed similarities per class"""
    class_to_similarities = {}

    for i, class_label in enumerate(class_labels):
        class_loader = get_class_data_loader(
            image_net_dir, batch_size=1, class_label=class_label
        )
        transformed_similarity = compute_transformed_similarity(
            class_loader, transformation, max_n_image_pairs, similarity_type
        )
        class_to_similarities[class_label] = transformed_similarity
        if i % 100 == 0:
            log.debug(
                f"transformed similarity {i} classes out of {len(class_labels)} complete"
            )
    return class_to_similarities


def run_transformed_similarities(
    image_net_dir: Path,
    class_labels: List[str],
    similarity_type: str,
    max_n_image_pairs: int = 100,
    small_angle: float = 5.0,
    large_angle: float = 90.0,
) -> Dict[str, float]:
    """Compares..."""
    small_rotation = partial(rotate, angle=small_angle)
    large_rotation = partial(rotate, angle=large_angle)

    small_rotation_similarities = compute_transformed_similarities(
        image_net_dir,
        class_labels,
        small_rotation,
        max_n_image_pairs,
        similarity_type,
    )

    large_rotation_similarities = compute_transformed_similarities(
        image_net_dir,
        class_labels,
        large_rotation,
        max_n_image_pairs,
        similarity_type,
    )

    avg_small = compute_class_avg(small_rotation_similarities)
    std_small = compute_class_std(small_rotation_similarities)

    avg_large = compute_class_avg(large_rotation_similarities)
    std_large = compute_class_std(large_rotation_similarities)

    result = {
        "avg for small rotation": avg_small,
        "avg for large rotation": avg_large,
        "std for small rotation": std_small,
        "std for large rotation": std_large,
    }
    return result


def get_class_labels(cfg: DictConfig, image_net_dir: Path):
    if not cfg.sanity_checks.run_on_sample_classes:
        # return full set of labels from directory
        return load_class_labels(image_net_dir)

    if image_net_dir.name == "val":
        class_labels = cfg.sanity_checks.sample_val_class_labels
    else:
        # for testing locally
        class_labels = cfg.sanity_checks.sample_train_class_labels
    return class_labels


def summarize_results(
    across_v_within_similarity,
    transformed_similarity,
    across_indep_transformed_similarity,
    across_pairwise_transformed_similarity,
) -> str:
    """Returns a LaTeX table with results"""
    df = pd.DataFrame(columns=["Comparison", "Average", "Std"])

    for rot_type in ["large", "small"]:
        df = df.append(
            {
                "Comparison": f"{rot_type} rotation",
                "Average": transformed_similarity[f"avg for {rot_type} rotation"],
                "Std": transformed_similarity[f"std for {rot_type} rotation"],
            },
            ignore_index=True,
        )

    for name in ["within", "across"]:
        df = df.append(
            {
                "Comparison": f"{name} class",
                "Average": across_v_within_similarity[f"avg {name}"],
                "Std": across_v_within_similarity[f"std {name}"],
            },
            ignore_index=True,
        )
        df = df.append(
            {
                "Comparison": f"{name} class pairwse random rotation",
                "Average": across_pairwise_transformed_similarity[f"avg {name}"],
                "Std": across_pairwise_transformed_similarity[f"std {name}"],
            },
            ignore_index=True,
        )
        df = df.append(
            {
                "Comparison": f"{name} class random rotation",
                "Average": across_indep_transformed_similarity[
                    f"avg transformed {name}"
                ],
                "Std": across_indep_transformed_similarity[f"std transformed {name}"],
            },
            ignore_index=True,
        )

    return df.to_latex(index=False)


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    image_net_dir = get_image_net_dir()
    class_labels = get_class_labels(cfg, image_net_dir)

    try:
        env = submitit.JobEnvironment()
        log.info(f"job id: {env.job_id}")
    except RuntimeError:
        warnings.warn("Not launched with submitit")

    across_v_within_similarity = run_within_v_across(
        image_net_dir,
        class_labels,
        cfg.sanity_checks.similarity_type,
        max_n_image_pairs=cfg.sanity_checks.max_n_image_pairs,
        max_class_label_combinations=cfg.sanity_checks.max_class_label_combinations,
        randomly_rotate=False,
        pairs_transform=None,
    )
    log.info(f"across vs. within similairty: {across_v_within_similarity}")

    transformed_similarity = run_transformed_similarities(
        image_net_dir,
        class_labels,
        cfg.sanity_checks.similarity_type,
        max_n_image_pairs=cfg.sanity_checks.max_n_image_pairs,
        small_angle=cfg.sanity_checks.small_angle,
        large_angle=cfg.sanity_checks.large_angle,
    )
    log.info(f"small versus large rotation similarity: {transformed_similarity}")

    across_indep_transformed_similarity = run_within_v_across_indep_transformed(
        image_net_dir,
        class_labels,
        cfg.sanity_checks.similarity_type,
        max_n_image_pairs=cfg.sanity_checks.max_n_image_pairs,
        max_class_label_combinations=cfg.sanity_checks.max_class_label_combinations,
    )
    log.info(
        f"across class independently transformed similarity: "
        f"{across_indep_transformed_similarity}"
    )

    across_pairwise_transformed_similarity = run_within_v_across_pairwise_transformed(
        image_net_dir,
        class_labels,
        cfg.sanity_checks.similarity_type,
        max_n_image_pairs=cfg.sanity_checks.max_n_image_pairs,
        max_class_label_combinations=cfg.sanity_checks.max_class_label_combinations,
    )
    log.info(
        f"across class pairwise transformed similarity: "
        f"{across_pairwise_transformed_similarity}"
    )

    summary = summarize_results(
        across_v_within_similarity,
        transformed_similarity,
        across_indep_transformed_similarity,
        across_pairwise_transformed_similarity,
    )
    log.info(f"summary for {cfg.sanity_checks.similarity_type} \n {summary}")


if __name__ == "__main__":
    main()
