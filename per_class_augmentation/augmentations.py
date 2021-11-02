"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import random
from PIL import Image, ImageEnhance, ImageOps
from typing import List, Optional
import numpy as np
import torchvision
import pandas as pd


class Augmentation:
    """Applies each augmentation with the given magnitude"""

    RANGES = {
        "shearX": np.linspace(0, 0.3, 10),
        "shearY": np.linspace(0, 0.3, 10),
        "translateX": np.linspace(0, 150 / 331, 10),
        "translateY": np.linspace(0, 150 / 331, 10),
        "rotate": np.linspace(0, 30, 10),
        "color": np.linspace(0.0, 0.9, 10),
        "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
        "solarize": np.linspace(256, 0, 10),
        "contrast": np.linspace(0.0, 0.9, 10),
        "sharpness": np.linspace(0.0, 0.9, 10),
        "brightness": np.linspace(0.0, 0.9, 10),
        "autocontrast": [0] * 10,
        "equalize": [0] * 10,
        "invert": [0] * 10,
        "rescale": list(range(10)),
    }

    FILLCOLOR = (128, 128, 128)

    NAME_TO_OP = {
        "shearX": lambda img, magnitude: img.transform(
            img.size,
            Image.AFFINE,
            (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC,
            fillcolor=Augmentation.FILLCOLOR,
        ),
        "shearY": lambda img, magnitude: img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC,
            fillcolor=Augmentation.FILLCOLOR,
        ),
        "translateX": lambda img, magnitude: img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=Augmentation.FILLCOLOR,
        ),
        "translateY": lambda img, magnitude: img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
            fillcolor=Augmentation.FILLCOLOR,
        ),
        "rotate": lambda img, magnitude: Augmentation.rotate_with_fill(img, magnitude),
        "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
            1 + magnitude * random.choice([-1, 1])
        ),
        "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
        "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
        "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
            1 + magnitude * random.choice([-1, 1])
        ),
        "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
            1 + magnitude * random.choice([-1, 1])
        ),
        "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
            1 + magnitude * random.choice([-1, 1])
        ),
        "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
        "equalize": lambda img, magnitude: ImageOps.equalize(img),
        "invert": lambda img, magnitude: ImageOps.invert(img),
        "rescale": lambda img, magnitude: rescale(img, magnitude),
    }

    def __init__(self, operation_name, magnitude_idx, uniformly_sample_magnitude=True):
        self.operation_name = operation_name
        self.magnitude_idx = magnitude_idx
        self.uniformly_sample_magnitude = uniformly_sample_magnitude

    def __hash__(self):
        """For comparing objects in a set"""
        return hash((self.operation_name, self.magnitude_idx))

    def __repr__(self) -> str:
        return f"{self.operation_name} with magnitude {self.magnitude_idx}"

    def __call__(self, img):
        op = Augmentation.NAME_TO_OP[self.operation_name]
        if self.uniformly_sample_magnitude:
            magnitude_idx = self.sample_magnitude_idx()
        else:
            magnitude_idx = self.magnitude_idx
        magnitude = Augmentation.RANGES[self.operation_name][magnitude_idx]
        transformed_img = op(img, magnitude)
        return transformed_img

    def sample_magnitude_idx(self):
        """Uniformly samples [0, magnitude index]"""
        return int(np.random.randint(self.magnitude_idx + 1))

    @property
    def name(self):
        return f"{self.operation_name} with magnitude {self.magnitude_idx}"

    # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
    @staticmethod
    def rotate_with_fill(img, magnitude):
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(
            rot, Image.new("RGBA", rot.size, (128,) * 4), rot
        ).convert(img.mode)


def to_scale(magnitude):
    """Returns a scale between [0.28, 1/.28].

    Args:
        magnitude (int): between 0 and 9 indicating magnitude.
            0 doesn't change scale
            1-5: zooms out. 6-9: zooms in
    """
    if magnitude < 0.0 or magnitude > 9.0:
        raise ValueError("magnitude must be within 0 and 9")
    zoom_out_step = 0.75 / 5.0
    zoom_in_step = (3.5 - 1.0) / 4.0

    magnitude_to_scale = {
        0: 1.0,
        1: 1.0 - zoom_out_step,
        2: 1.0 - 2 * zoom_out_step,
        3: 1.0 - 3 * zoom_out_step,
        4: 1.0 - 4 * zoom_out_step,
        5: 0.25,
        6: 1.0 + zoom_in_step,
        7: 1.0 + 2 * zoom_in_step,
        8: 1.0 + 3 * zoom_in_step,
        9: 3.5,
    }

    return magnitude_to_scale[int(magnitude)]


def rescale(img, magnitude):
    """Scales using pytorch"""
    scale = to_scale(magnitude)
    return torchvision.transforms.functional.affine(img, 0.0, (0.0, 0.0), scale, 0.0)


class MultipleAugmentations:
    def __init__(self, augmentations: List[Augmentation], probabilities: List[float]):
        self.augmentations = augmentations
        self.probabilities = probabilities

    def __call__(self, img) -> Image.Image:
        for augmentation, prob in zip(self.augmentations, self.probabilities):
            if prob > random.random():
                img = augmentation(img)
        return img

    def __len__(self) -> int:
        return len(self.augmentations)

    def __repr__(self):
        augmentation_names = [str(a) for a in self.augmentations]
        return (
            f"{len(augmentation_names)} augmentations: {augmentation_names}"
            f"\nprobabilities: {self.probabilities}"
        )


class TopWeightedAugmentations:
    """
    Computes the top augmentations per class or across classes

    Args:
        num_transforms: num of transforms to apply per/across classes
        transform_dir: directory for dataframe containing transform changes
        similarity_type: "resnet18_no_aug" or "resnet18" trained with standard augs
        transform_prob: dictionary containing distribution or fixed value
        top_per_class: if true, top transform is per class. Else top across all classes is selected.
        min_prop_boosted_filter: threshold for minimum proportion boosted needed to keep a transformation
    """

    def __init__(
        self,
        num_transforms=2,
        transform_dir: str = "",
        similarity_type: str = "resnet18_no_aug",
        transform_prob: dict = dict({"dist": "fixed", "fixed_prob": 0.1}),
        top_per_class: bool = True,
        top_transform_ranking: str = "proportion_boosted",
        min_prop_boosted_filter: Optional[float] = 0.4,
        min_perc_change_per_class_filter: Optional[float] = 0.0,
    ):
        self.num_transforms = num_transforms
        self.transform_path = (
            f"{transform_dir}/similarity_search_{similarity_type}/"
            f"single_transform_boosts_train.csv"
        )
        self.top_per_class = top_per_class
        self.top_transform_ranking = top_transform_ranking
        self.transform_prob = transform_prob
        self.min_prop_boosted_filter = min_prop_boosted_filter
        self.min_perc_change_per_class_filter = min_perc_change_per_class_filter

        self.df = self.load_top_transform_df()

        if top_per_class:
            self.top_transforms_df = self.group_top_per_class()
        else:
            self.top_transforms_df = self.group_top_across_classes()

    def load_top_transform_df(self) -> pd.DataFrame:
        """Returns dataframe of transformations"""
        df = pd.read_csv(self.transform_path, index_col=0)
        df["weighted_boost"] = df["avg_percent_boost"] * df["proportion_boosted"]
        return df

    def group_top_per_class(self) -> pd.DataFrame:
        """Ranks top transforms per class by avg_percent_similarity_change."""
        return self.df.groupby(["class_label"]).apply(
            lambda x: x.nlargest(
                self.num_transforms, columns=[self.top_transform_ranking]
            )
        )

    def group_top_across_classes(self) -> pd.DataFrame:
        """Ranks top transforms by avg_percent_similarity_change."""
        return (
            self.df.groupby(["transform_name"])
            .mean()
            .nlargest(self.num_transforms, columns=[self.top_transform_ranking])
        ).reset_index()

    @staticmethod
    def transform_names_to_augmentations(
        transform_names: List[str],
    ) -> List[Augmentation]:
        operation_names = [parse_operation_name(name) for name in transform_names]
        magnitudes = [parse_magnitude(name) for name in transform_names]
        augmentations = [
            Augmentation(name, magnitude)
            for name, magnitude in zip(operation_names, magnitudes)
        ]
        return augmentations

    def compute_probabilities(self, df: pd.DataFrame):
        num_transforms = len(df)
        if self.transform_prob["dist"] == "fixed":
            val = self.transform_prob["fixed_prob"]
            probabilities = uniform_dist(val=val, size=num_transforms)
        elif self.transform_prob["dist"] == "uniform":
            probabilities = uniform_dist(size=num_transforms)
        elif self.transform_prob["dist"] == "weighted_boost":
            if len(df) == 1:
                probabilities = [0.5]
            else:
                probabilities = softmax(df["weighted_boost"].values)
        else:
            raise NotImplementedError(f"{self.transform_prob=} not supported")
        return probabilities

    def filter_top_by_class_label(self, class_label: str):
        """Filters dataframe for given class label"""
        df = self.top_transforms_df
        if self.top_per_class:
            df = df[df["class_label"] == class_label]
        return df

    def filter_top_by_prop_boosted(self, df: pd.DataFrame):
        """Filters top transformations that don't boost more than"""
        if self.min_prop_boosted_filter is None:
            return df
        return df[df["proportion_boosted"] > self.min_prop_boosted_filter]

    def filter_by_min_per_change(self, df: pd.DataFrame):
        """Filters top transformations with average perc change below threshold"""
        if not self.top_per_class or (self.min_perc_change_per_class_filter is None):
            return df
        return df[
            df["avg_percent_similarity_change"] >= self.min_perc_change_per_class_filter
        ]

    def get_top_transforms(self, class_label=None):
        """Returns top transformation across or per class.

        Args:
            class_label: class for which to return top transforms
        """
        df = self.filter_top_by_class_label(class_label)
        df = self.filter_top_by_prop_boosted(df)
        df = self.filter_by_min_per_change(df)
        # no transformations match criteria
        if len(df) == 0:
            return MultipleAugmentations([], [])

        transform_names = df.transform_name.values.tolist()
        augmentations = self.transform_names_to_augmentations(transform_names)
        probabilities = self.compute_probabilities(df)
        transforms = MultipleAugmentations(augmentations, probabilities)
        return transforms


def parse_magnitude(transform_name: str) -> int:
    return int(transform_name.split()[-1])


def parse_operation_name(transform_name: str) -> str:
    return transform_name.split()[0]


def uniform_dist(val=None, size=10) -> List[float]:
    """Returns vector of given size with entries 1 / size"""
    if val is None:
        val = 1.0 / size
    return (np.ones(size) * val).tolist()


def softmax(weights):
    """Normalizes weights via softmax"""
    return (np.exp(weights) / np.exp(weights).sum()).tolist()