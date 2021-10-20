import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import affine


def get_all_angles():
    """Divides -180, 180 into [0, small negative, small positive, larger...]"""
    return sorted(
        np.unique(np.concatenate([np.linspace(0, 180, 5), np.linspace(-135, 0, 6)])),
        key=lambda x: abs(x),
    )


def get_shear_angles():
    """Returns angles [0, small negatie, small positive, ... 90.0].
    Note Shear 90 degrees is empty
    """
    angles = np.unique(
        np.concatenate([np.linspace(0, 91.0, 5), np.linspace(0, -89.0, 6)])
    )
    return sorted(angles, key=lambda x: abs(x))


class Transformation:
    """Applies transformation to PIL or torch tensor"""

    RANGES = {
        "shearX": get_shear_angles(),
        "shearY": get_shear_angles(),
        "translateX": np.linspace(0, 224, 10),
        "translateY": np.linspace(0, 224, 10),
        "rotate": get_all_angles(),
        "scale_zoom_out": np.linspace(1.0, 0.28, 10),
        "scale_zoom_in": np.linspace(1.0, 3.5, 10),
        "posterize": np.round(np.linspace(8, 0, 10), 0).astype(int),
        "decrease_contrast": np.linspace(1.0, 0.0, 10),
        "increase_contrast": np.linspace(1.0, 2.0, 10),
        "solarize": np.linspace(1.0, 0, 10),
        "decrease_brightness": np.linspace(1.0, 0.0, 10),
        "increase_brightness": np.linspace(1.0, 2.0, 10),
    }

    NAME_TO_FUNC = {
        "shearX": "affine",
        "shearY": "affine",
        "translateX": "affine",
        "translateY": "affine",
        "rotate": "affine",
        "scale_zoom_in": "affine",
        "scale_zoom_out": "affine",
        "posterize": "posterize",
        "increase_brightness": "adjust_brightness",
        "decrease_brightness": "adjust_brightness",
        "increase_contrast": "adjust_contrast",
        "decrease_contrast": "adjust_contrast",
    }

    def __init__(self, operation_name, magnitude_idx):
        self.operation_name = operation_name
        self.magnitude_idx = magnitude_idx

    def __call__(self, img):
        magnitude = self.RANGES[self.operation_name][self.magnitude_idx]
        if self.operation_name in self.NAME_TO_FUNC:
            func_name = self.NAME_TO_FUNC[self.operation_name]
            return getattr(self, f"apply_{func_name}")(img, magnitude)
        return getattr(torchvision.transforms.functional, self.operation_name)(
            img, magnitude
        )

    def apply_posterize(self, img, magnitude):
        """Applies posterize after converting img to unit8"""
        if not type(img) is torch.Tensor:
            return torchvision.transforms.functional.posterize(img, magnitude)

        img = (img * 100).to(torch.uint8)
        img = torchvision.transforms.functional.posterize(img, magnitude).float()
        img = img / 100.0
        return img

    def apply_adjust_contrast(self, img, magnitude):
        return torchvision.transforms.functional.adjust_contrast(img, magnitude)

    def apply_adjust_brightness(self, img, magnitude):
        return torchvision.transforms.functional.adjust_brightness(img, magnitude)

    def apply_affine(self, img, magnitude):
        magnitudes = {
            "shearX": 0.0,
            "shearY": 0.0,
            "translateX": 0.0,
            "translateY": 0.0,
            "rotate": 0.0,
            "scale": 1.0,
        }

        magnitudes[self.operation_name] = magnitude
        if "scale" in self.operation_name:
            magnitudes["scale"] = magnitude

        img_transformed = affine(
            img,
            shear=(magnitudes["shearX"], magnitudes["shearY"]),
            translate=(magnitudes["translateX"], magnitudes["translateY"]),
            angle=magnitudes["rotate"],
            scale=magnitudes["scale"],
            interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR,
        )
        return img_transformed


TRANSFORMATION_NAMES = list(Transformation.RANGES.keys())
