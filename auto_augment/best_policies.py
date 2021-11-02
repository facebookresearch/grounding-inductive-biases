"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Classes containing best augmentation policies based on 
https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
"""

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import torch
import torchvision


class Transformation:
    def __init__(self, operation_name, magnitude_idx, magnitude, operation):
        self.operation_name = operation_name
        self.magnitude_idx = magnitude_idx
        self.magnitude = magnitude
        self.operation = operation

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.operation_name != other.operation_name:
            return False
        if self.magnitude != other.magnitude:
            return False
        return True

    def __hash__(self):
        """For comparing objects in a set"""
        return hash((self.operation_name, self.magnitude_idx))

    @property
    def name(self):
        return f"{self.operation_name} with magnitude {self.magnitude_idx}"

    def apply_single(self, img):
        """Applies transformation with probability 1. Returns a tensor image"""
        if torch.is_tensor(img):
            img = to_image(img.detach().clone())
        img = self.operation(img, self.magnitude)
        # careful to_tensor returns values between 0 and 1, but pil_to_tensor does not!
        image_tensor = torchvision.transforms.functional.to_tensor(img).type(
            torch.float
        )
        return image_tensor

    def apply(self, images):
        """Applies transformation with probability 1. Returns a tensor image"""
        transformed_images = []
        for image in images:
            transformed_image = self.apply_single(image)
            transformed_images.append(transformed_image)
        return torch.stack(transformed_images)

    def __repr__(self) -> str:
        return f"{self.name} with magnitude {self.magnitude_idx}"


class ImageNetPolicy:
    """Randomly choose one of the best 24 Sub-policies on ImageNet.

    Example:
    >>> policy = ImageNetPolicy()
    >>> transformed = policy(image)

    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     ImageNetPolicy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
        ]

    def get_unique_single_transformations(self):
        return policies_to_single_transformations(self.policies)

    def get_unique_subpolicies(self):
        subpolicy_names = set()
        unique_subpolicies = []

        for subpolicy in self.policies:
            if subpolicy.name not in subpolicy_names:
                unique_subpolicies.append(subpolicy)
                subpolicy_names.add(subpolicy.name)
        return unique_subpolicies

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """Randomly choose one of the best 25 Sub-policies on CIFAR10.

    Example:
    >>> policy = CIFAR10Policy()
    >>> transformed = policy(image)

    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     CIFAR10Policy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
        ]

    def get_transformations(self):
        return policies_to_single_transformations(self.policies)

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """Randomly choose one of the best 25 Sub-policies on SVHN.

    Example:
    >>> policy = SVHNPolicy()
    >>> transformed = policy(image)

    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     SVHNPolicy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),
            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor),
        ]

    def get_transformations(self):
        return policies_to_transformations(self.policies)

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(
        self,
        p1,
        operation1,
        magnitude_idx1,
        p2,
        operation2,
        magnitude_idx2,
        fillcolor=(128, 128, 128),
    ):
        self.operation1_name, self.operation2_name = operation1, operation2
        self.magnitude_idx1, self.magnitude_idx2 = magnitude_idx1, magnitude_idx2

        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
            "rescale": list(range(10)),
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
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

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __repr__(self):
        return (
            f"{self.operation1_name} with magnitude {self.magnitude_idx1}) (original prob {self.p1})"
            f"{self.operation2_name} with magnitude {self.magnitude_idx2}) (original prob {self.p2})"
        )

    def __hash__(self):
        """For comparing objects in a set"""
        return hash(self.name)

    @property
    def name(self):
        op_names = f"{self.operation1_name} and {self.operation2_name}"
        magnitudes = f"magnitudes {self.magnitude_idx1}, {self.magnitude_idx2}"
        return f"{op_names} with {magnitudes}"

    def average_prob_applied(self):
        return np.mean([self.p1, self.p2])

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img

    def apply_single(self, img):
        """Applies transformations with probability 1. Returns a tensor image"""
        if torch.is_tensor(img):
            img = to_image(img)
        img = self.operation1(img, self.magnitude1)
        img = self.operation2(img, self.magnitude2)
        image_tensor = torchvision.transforms.functional.to_tensor(img).to(torch.float)
        return image_tensor

    def apply(self, images):
        """Applies transformations with probability 1. Returns a tensor image"""
        transformed_images = []
        for image in images:
            transformed_image = self.apply_single(image)
            transformed_images.append(transformed_image)
        return torch.stack(transformed_images)


def policies_to_single_transformations(policies):
    """Returns list of transformations with their associated propabilities"""
    transformations = set()

    for sub_policy in policies:
        name1, name2 = sub_policy.operation1_name, sub_policy.operation2_name
        transformation1 = Transformation(
            name1,
            sub_policy.magnitude_idx1,
            sub_policy.magnitude1,
            sub_policy.operation1,
        )
        transformation2 = Transformation(
            name2,
            sub_policy.magnitude_idx2,
            sub_policy.magnitude2,
            sub_policy.operation2,
        )
        transformations.add(transformation1)
        transformations.add(transformation2)
    return list(transformations)


def to_image(image_tensor: torch.Tensor):
    """Converts input into a PIL image"""
    with torch.no_grad():
        to_pil = torchvision.transforms.ToPILImage()
        return to_pil(image_tensor.detach().squeeze())


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
