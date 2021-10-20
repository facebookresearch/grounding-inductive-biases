from equivariance_measure.embedding_alignments import compute_cos_distance
import json
import os
import torch
import timm
import numpy as np
import pytorch_lightning as pl
from torchvision.models import resnet18
from equivariance_measure import transformations
from tdigest import TDigest
from collections import defaultdict, OrderedDict
from typing import List, Callable, Dict
from functools import cache


class TransformInvarianceDigest:
    """Stores percentiles for relative invariance"""

    def __init__(
        self,
        transform_name: str,
        distance_type: str = "cos_similarity",
        data_stage: str = "train",
    ):
        self.transform_name = transform_name
        self.distance_type = distance_type
        self.data_stage = data_stage
        # magnitude -> digest
        self.magnitude_to_invariance = defaultdict(TDigest)
        self.magnitude_to_baseline = defaultdict(TDigest)
        self.magnitude_to_transformed_distance = defaultdict(TDigest)

    def batch_update(
        self,
        magnitude: int,
        baseline: np.array,
        transformed_distance: np.array,
        invariance: np.array,
    ):
        self.magnitude_to_baseline[str(magnitude)].batch_update(baseline)
        self.magnitude_to_transformed_distance[str(magnitude)].batch_update(
            transformed_distance
        )
        self.magnitude_to_invariance[str(magnitude)].batch_update(invariance)

    @property
    def magnitudes(self) -> List[str]:
        return list(self.magnitude_to_invariance.keys())

    def get_percentile(self, magntiude: int, percentile: float):
        return self.magnitude_to_invariance[str(magntiude)].percentile(percentile)

    def get_baseline_percentile(self, magntiude: int, percentile: float):
        return self.magnitude_to_baseline[str(magntiude)].percentile(percentile)

    def get_transformed_distance_percentile(self, magntiude: int, percentile: float):
        return self.magnitude_to_transformed_distance[str(magntiude)].percentile(
            percentile
        )

    @staticmethod
    def serialize_digest(magnitude_to_digest: Dict[str, TDigest]) -> dict:
        magnitude_to_distances_dict = dict()

        for magnitude, digest in magnitude_to_digest.items():
            magnitude_to_distances_dict[magnitude] = digest.to_dict()
        return magnitude_to_distances_dict

    def serialize(self):
        magnitude_to_baseline_dict = self.serialize_digest(self.magnitude_to_baseline)
        magnitude_to_transformed_distance_dict = self.serialize_digest(
            self.magnitude_to_transformed_distance
        )
        magnitude_to_invariance_dict = self.serialize_digest(
            self.magnitude_to_invariance
        )
        summary_dict = {
            "baseline": magnitude_to_baseline_dict,
            "transformed_distance": magnitude_to_transformed_distance_dict,
            "invariance": magnitude_to_invariance_dict,
        }
        return summary_dict

    def load_from_dict(self, type_to_magnitude_to_digest_dict: dict):
        """Loads results"""
        for dist_type in type_to_magnitude_to_digest_dict:
            for magnitude, digest_dict in type_to_magnitude_to_digest_dict[
                dist_type
            ].items():
                magnitude_to_dist = getattr(self, f"magnitude_to_{dist_type}")
                magnitude_to_dist[magnitude].update_from_dict(digest_dict)


class EmbeddingDistanceModule(pl.LightningModule):
    def __init__(
        self,
        num_classes=1000,
        pretrained=True,
        results_dir=None,
        data_stage="train",
        transform_name="shearX",
        model_pth=None,
        prefix="",
    ):
        super().__init__()

        self.results_dir = results_dir
        self.data_stage = data_stage
        self.model_pth = model_pth
        self.model = self.get_model_without_last_layer(num_classes, pretrained)
        self.prefix = prefix
        self.transform_name = transform_name

        # embeddings matrix Z
        self.embeddings_matrix = []
        # transformed embeddings matrix Z
        self.magnitude_to_transformed_embeddings_matrix = defaultdict(list)

        self.cos_distance_digest = TransformInvarianceDigest(
            transform_name,
            distance_type=f"{prefix}cos_similarity",
            data_stage=data_stage,
        )
        self.l2_digest = TransformInvarianceDigest(
            transform_name, distance_type=f"{prefix}l2", data_stage=data_stage
        )

    def get_model_without_last_layer(
        self, num_classes: int, pretrained: bool
    ) -> torch.nn.Sequential:
        model = get_resnet18(num_classes, pretrained)
        if self.model_pth:
            model = self.load_pretrained_ddp(model)
        embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        return embedding_model

    def load_pretrained_ddp(self, model) -> torch.nn.Module:
        state_dict = torch.load(self.model_pth)["state_dict"]
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        return model

    def forward(self, x):
        return self.model(x).squeeze()

    def test_step(self, batch, batch_idx):
        x, labels = batch
        z = self(x)

        for magnitude_idx in range(10):
            transform = transformations.Transformation(
                self.transform_name, magnitude_idx
            )
            x_t = transform(x)
            z_t = self(x_t)
            self.magnitude_to_transformed_embeddings_matrix[magnitude_idx].append(z_t)
        self.embeddings_matrix.append(z)

    def on_test_start(self):
        self.cleanup_results()
        return super().on_test_start()

    def cleanup_results(self):
        # cleanup old results
        self.embeddings_matrix = []
        self.magnitude_to_transformed_embeddings_matrix = defaultdict(list)
        self.cos_distance_digest = TransformInvarianceDigest(
            self.transform_name,
            distance_type=f"{self.prefix}cos_similarity",
            data_stage=self.data_stage,
        )
        self.l2_digest = TransformInvarianceDigest(
            self.transform_name,
            distance_type=f"{self.prefix}l2",
            data_stage=self.data_stage,
        )

    def get_distance_func(self, distance_type: str) -> Callable:
        if distance_type == "cos_distance":
            return compute_cos_distance
        elif distance_type == "l2":
            l2_dist = torch.nn.PairwiseDistance()
            return l2_dist
        raise ValueError(f"{distance_type} not supported")

    def compute_relative_invariance(
        self,
        Z: torch.Tensor,
        magnitude_to_Z_t: Dict[int, torch.Tensor],
        distance_type="cos_distance",
    ):
        """Computes dist(Z, shuffled(Z_t)) - dist(Z, Z_t)"""
        dist = self.get_distance_func(distance_type)
        for magnitude_idx, Z_t in magnitude_to_Z_t.items():
            perms = torch.randperm(Z_t.shape[0])
            Z_t_shuffled = Z_t[perms, :]
            baseline = dist(Z, Z_t_shuffled)
            transformed_distance = dist(Z, Z_t)
            # for numerical stability
            epsilon = 1e-8
            invariance = torch.divide(
                baseline - transformed_distance, baseline + epsilon
            )
            digest = getattr(self, f"{distance_type}_digest")
            digest.batch_update(
                str(magnitude_idx),
                baseline.cpu().numpy(),
                transformed_distance.cpu().numpy(),
                invariance.cpu().numpy(),
            )

    def create_magnitude_to_Z_t(self) -> Dict[int, torch.Tensor]:
        """Turns list of tensors into a single d_matrix.
        d_matrix shape is [n_samples, embedding_size]
        """
        magnitude_to_Z_t = dict()
        for magnitude_idx in range(10):
            Z_t = torch.cat(
                self.magnitude_to_transformed_embeddings_matrix[magnitude_idx]
            )
            magnitude_to_Z_t[magnitude_idx] = Z_t
        return magnitude_to_Z_t

    def on_test_end(self):
        Z = torch.cat(self.embeddings_matrix)
        magnitude_to_Z_t = self.create_magnitude_to_Z_t()
        self.compute_relative_invariance(
            Z, magnitude_to_Z_t, distance_type="cos_distance"
        )
        self.compute_relative_invariance(Z, magnitude_to_Z_t, distance_type="l2")
        return super().on_test_end()


class EmbeddingDistanceViTModule(EmbeddingDistanceModule):
    def __init__(
        self,
        num_classes=1000,
        pretrained=True,
        results_dir=None,
        data_stage="train",
        transform_name="shearX",
        prefix="",
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            results_dir=results_dir,
            transform_name=transform_name,
            data_stage=data_stage,
            prefix=prefix,
        )
        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)

    def forward(self, x):
        return self.model.forward_features(x)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        z = self(x)

        for magnitude_idx in range(10):
            transform = transformations.Transformation(
                self.transform_name, magnitude_idx
            )
            x_t = transform(x)
            z_t = self(x_t)
            self.magnitude_to_transformed_embeddings_matrix[magnitude_idx].append(
                z_t.cpu()
            )
        self.embeddings_matrix.append(z.cpu())


class InvariancesDigest:
    def __init__(
        self,
        distance_type: str,
        data_stage: str,
        prefix: str = "",
        model_name: str = "",
    ):
        self.distance_type = distance_type
        self.data_stage = data_stage
        self.transform_to_invariance_digest = dict()
        self.prefix = prefix
        self.model_name = model_name

    @property
    def magnitudes(self) -> List[int]:
        return list(str(x) for x in range(10))

    @property
    def transform_names(self) -> List[str]:
        return list(self.transform_to_invariance_digest.keys())

    def get_percentile(self, transform: str, magnitude: str, percentile: int) -> float:
        return self.transform_to_invariance_digest[transform].get_percentile(
            str(magnitude), percentile
        )

    def get_baseline_percentile(
        self, transform: str, magnitude: str, percentile: int
    ) -> float:
        return self.transform_to_invariance_digest[transform].get_baseline_percentile(
            str(magnitude), percentile
        )

    def update(self, model: EmbeddingDistanceModule):
        """Extracts digests from each model"""
        self.transform_to_invariance_digest[model.transform_name] = getattr(
            model, f"{self.distance_type}_digest"
        )

    def save(self, results_dir: str):
        file_name = (
            f"{self.prefix}{self.distance_type}_invariance_{self.data_stage}.json"
        )
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, file_name)
        transform_to_invariance = {
            k: v.serialize() for k, v in self.transform_to_invariance_digest.items()
        }
        with open(path, "w") as f:
            json.dump(transform_to_invariance, f)

    def load(self, results_dir: str):
        file_name = (
            f"{self.prefix}{self.distance_type}_invariance_{self.data_stage}.json"
        )

        path = os.path.join(results_dir, file_name)
        with open(path, "r") as f:
            invariances = json.load(f)
        transform_to_invariance = dict()

        for transform, digest_dict in invariances.items():
            invariances_digest = TransformInvarianceDigest(
                transform, data_stage=self.data_stage, distance_type=self.distance_type
            )
            invariances_digest.load_from_dict(digest_dict)
            transform_to_invariance[transform] = invariances_digest

        self.transform_to_invariance_digest = transform_to_invariance


@cache
def get_resnet18(num_classes: int, pretrained: bool):
    return resnet18(num_classes=num_classes, pretrained=pretrained)
