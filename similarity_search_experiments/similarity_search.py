"""
Computes the similarity between transformed image_1 and image_2
with the subpolicy transformations from AutoAugment's best ImageNet augmentation policies
"""


import torch
import numpy as np
import pandas as pd
import logging

from torch import Tensor
from dataclasses import dataclass
from more_itertools import grouper
from pathlib import Path
from typing import List, Tuple, Union, Dict, Optional
from image_similarity import load_pairs
from image_similarity.similarity import embedding_similarity
from auto_augment.new_policies import ImageNetPlusGeometric
from auto_augment.best_policies import (
    Transformation,
    SubPolicy,
)


log = logging.getLogger(__name__)
# log.setLevel("DEBUG")


@dataclass
class Boost:
    """Stores similarity changes for a transformation"""

    transform_name: str
    # transformed - original similarities
    similarity_changes: np.ndarray
    original_similarities: np.ndarray

    def __post_init__(self):
        """Verifies similarity_changes is a numpy array"""
        if not type(self.similarity_changes) is np.ndarray:
            current_type = type(self.similarity_changes)
            raise ValueError(
                f"similarity_changes must be a np.ndarray not {current_type}"
            )
        if not self.is_same_size_similarity():
            raise ValueError(
                f"Similarities are different sizes original "
                f"{self.original_similarities.shape=} {self.similarity_changes.shape=}"
            )

    def __repr__(self):
        return f"{self.similarity_changes.size=} {self.original_similarities.size=}"

    def is_same_size_similarity(self):
        """Verifies similarity changes and originals are same size"""
        if self.original_similarities.shape != self.similarity_changes.shape:
            return False
        return True

    @property
    def boosts(self):
        """Returns positive changes in similarities only"""
        return self.similarity_changes[self.similarity_changes > 0]

    @property
    def boost_percent_change(self):
        originials = self.original_similarities[self.similarity_changes > 0]
        return self.boosts / originials

    @property
    def percent_change(self):
        return self.similarity_changes / self.original_similarities

    @property
    def decreases(self):
        """Returns decreases in similarities only"""
        return self.similarity_changes[self.similarity_changes < 0]

    @property
    def proportion_boosted(self):
        prop = float(len(self.boosts)) / len(self.similarity_changes)
        return prop


def compute_class_similarity_boosts(
    class_data_loader: torch.utils.data.DataLoader,
    transformations: Union[List[Transformation], List[SubPolicy]],
    max_n_image_pairs: int,
    similarity_type: str,
) -> List[Boost]:

    transform_to_changes: Dict[str, List[float]] = dict()
    transform_to_originals: Dict[str, List[float]] = dict()

    # check last batch is dropped (for evenly sized pairs)
    assert class_data_loader.drop_last, "drop_last should be True"

    pairs_count = 0

    for image_batch, _ in class_data_loader:
        # stop after max_n_image_pairs
        if pairs_count >= max_n_image_pairs:
            break
        images_1, images_2 = split_into_pairs(image_batch)
        similarities = embedding_similarity(
            images_1, images_2, model_type=similarity_type
        )
        pairs_count += similarities.size

        for i, transform in enumerate(transformations):
            images_2_transformed = transform.apply(images_2)
            transformed_similarities = embedding_similarity(
                images_1, images_2_transformed, model_type=similarity_type
            )
            changes = (transformed_similarities - similarities).tolist()
            transform_to_changes = extend_dict(
                transform_to_changes, transform.name, changes
            )
            transform_to_originals = extend_dict(
                transform_to_originals, transform.name, similarities.tolist()
            )

    boosts = [
        Boost(
            name,
            np.array(transform_to_changes[name]),
            np.array(transform_to_originals[name]),
        )
        for name in transform_to_changes
    ]
    return boosts


def extend_dict(d, k, values):
    """Extends dictionary with values in a list"""
    original = []
    if k in d:
        original = d[k]
    new = original + values
    d[k] = new
    return d


def split_into_pairs(image_batch: Tensor) -> Tuple[Tensor, Tensor]:
    """Splits a batch of images in two for comparing pairs"""
    batch_size = image_batch.shape[0]
    if batch_size % 2 != 0:
        raise ValueError(
            f"image_batch size {image_batch.shape} is not divisible by two"
        )
    images_1, images_2 = (
        image_batch[: batch_size // 2, :],
        image_batch[batch_size // 2 :, :],
    )
    return images_1, images_2


def compute_similarity_boosts(
    class_data_loaders: List[torch.utils.data.DataLoader],
    class_labels: List[str],
    transformations: Union[List[Transformation], List[SubPolicy]],
    max_n_image_pairs: int,
    similarity_type: str,
) -> pd.DataFrame:
    """
    Computes the change in similarity each transformation brings per class.
    """
    df = pd.DataFrame()

    for i, (label, class_data_loader) in enumerate(
        zip(class_labels, class_data_loaders)
    ):
        boosts = compute_class_similarity_boosts(
            class_data_loader,
            transformations,
            max_n_image_pairs,
            similarity_type,
        )
        class_df = boosts_to_dataframe(boosts, label)
        df = df.append(class_df)
        log_freq = float(len(class_labels)) // 10 + 1
        if i % (log_freq) == 0:
            log.info(f"{i} out of {len(class_labels)} classes complete")
    return df


def get_data_loaders(
    class_labels: List[str],
    image_net_dir: Path,
    batch_size: int,
) -> List[torch.utils.data.DataLoader]:

    data_loaders = []

    for label in class_labels:
        class_data_loader = load_pairs.get_class_data_loader(
            image_net_dir,
            batch_size=batch_size,
            class_label=label,
            drop_last=True,
        )
        data_loaders.append(class_data_loader)
    return data_loaders


def boosts_to_dataframe(boosts: List[Boost], class_label: str) -> pd.DataFrame:
    """Aggregates boosts in a pandas dataframe"""
    fields = ["similarity_changes", "boosts", "decreases"]
    results = dict()

    for field in fields:
        results[f"avg_{field[:-1]}"] = [
            np.mean(getattr(boost, field)) for boost in boosts
        ]
        results[f"std_{field[:-1]}"] = [
            np.std(getattr(boost, field)) for boost in boosts
        ]

    # add percentages
    results["avg_percent_similarity_change"] = [
        np.mean(boost.percent_change) for boost in boosts
    ]

    results["avg_percent_boost"] = [
        np.mean(boost.boost_percent_change) for boost in boosts
    ]
    # add proportion
    results["proportion_boosted"] = [boost.proportion_boosted for boost in boosts]
    results["num_boosted"] = [boost.boosts.size for boost in boosts]
    results["total_num_pairs"] = [boost.similarity_changes.size for boost in boosts]

    df = pd.DataFrame(results)
    df["transform_name"] = [boost.transform_name for boost in boosts]
    df["class_label"] = class_label
    return df


def save_boosts(df: pd.DataFrame, path: Path, name: str) -> None:
    """Saves boots in given path"""
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / f"{name}.csv")


def select_partition(
    labels: List[str], num_partitions: int, partition_index: int
) -> List[str]:
    if len(labels) % num_partitions != 0:
        raise ValueError(f"{num_partitions=} not divisible by {len(labels)=}")
    partition_size = len(labels) / num_partitions
    partitions = list(grouper(labels, int(partition_size)))
    return partitions[partition_index]


def main(
    class_labels: Optional[List[str]] = None,
    similarity_type: str = "resnet18",
    batch_size: int = 32,
    max_n_image_pairs: int = 50,
    use_val: bool = True,
    parallelize_classes: bool = False,
    num_partitions: int = 10,
    partition_index: int = 0,
    run_on_subpolicies: bool = False,
    save_path: Optional[Path] = None,
):
    """Runs similarity search on given class labels.
    If class_labels is None, runs on all of image net.

    Args:
        use_val (bool): runs analysis on the validation dataset
        parallelize_classes (bool): if True, class labels are partition into num_partitions.
            Similarity search runs on partition index.
        run_on_subpolicies (bool): determines whether analysis is for subpolicies or single transformations

    """
    image_net_policies = ImageNetPlusGeometric()
    image_net_dir = load_pairs.get_image_net_dir(use_val=use_val)

    log.info(f"image_net_dir {image_net_dir}")

    if class_labels is None:
        class_labels = load_pairs.load_class_labels(image_net_dir)

    if parallelize_classes:
        class_labels = select_partition(class_labels, num_partitions, partition_index)

    log.info(f"running with {len(class_labels)=}")

    data_loaders = get_data_loaders(class_labels, image_net_dir, batch_size)

    if run_on_subpolicies:
        transformations = image_net_policies.get_unique_subpolicies()
    else:
        transformations = image_net_policies.get_unique_single_transformations()

    boosts_df = compute_similarity_boosts(
        data_loaders,
        class_labels,
        transformations,
        max_n_image_pairs,
        similarity_type,
    )

    if save_path:
        data_type = "val" if use_val else "train"
        transform_type = "subpolicy" if run_on_subpolicies else "single_transform"
        save_boosts(
            boosts_df,
            save_path,
            f"{transform_type}_boosts_{data_type}",
        )


if __name__ == "__main__":
    tmp_dir = Path("~/Desktop/tmp").expanduser()
    main(save_path=tmp_dir)
