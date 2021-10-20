import json
import os
import pytorch_lightning as pl
import torch
import timm
from torch import Tensor
from torchvision.models import resnet18
from equivariance_measure import transformations
import torch.nn.functional as F
from tdigest import TDigest
from collections import defaultdict, OrderedDict
from typing import Dict, List


class TransformAlignmentsDigest:
    def __init__(self, transform_name: str, data_stage: str = "train"):
        self.magnitude_to_digest = defaultdict(TDigest)
        self.transform_name = transform_name
        self.data_stage = data_stage

    def batch_update(self, magnitude_to_alignment):
        for magnitude, alignments in magnitude_to_alignment.items():
            if isinstance(alignments, Tensor):
                alignments = alignments.cpu().numpy()
            self.magnitude_to_digest[magnitude].batch_update(alignments)

    def serialize(self) -> dict:
        alignments_dict = dict()
        for magnitude, digest in self.magnitude_to_digest.items():
            alignments_dict[magnitude] = digest.to_dict()
        return alignments_dict

    def load_from_dict(self, magnitude_to_centroids):
        for magnitude, centroids in magnitude_to_centroids.items():
            self.magnitude_to_digest[magnitude].update_from_dict(centroids)


class EmbeddingAlignmentModule(pl.LightningModule):
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
        self.prefix = prefix
        self.transform_name = transform_name
        self.model = self.get_model_without_last_layer(num_classes, pretrained)

        # stores z - z_t
        self.magnitude_to_diff = defaultdict(list)
        self.magnitude_to_shuffled_diff = None
        self.alignments = None
        self.alignments_digest = None

    def get_model_without_last_layer(
        self, num_classes: int, pretrained: bool
    ) -> torch.nn.Sequential:
        model = resnet18(num_classes=num_classes, pretrained=pretrained)
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
            d = z - z_t
            self.magnitude_to_diff[magnitude_idx].append(d.cpu())

    def on_test_start(self):
        # cleanup old results
        self.magnitude_to_diff = defaultdict(list)
        return super().on_test_start()

    def create_d_matrix(self):
        """Turns list of tensors into a single d_matrix.
        d_matrix shape is [n_samples, embedding_size]
        """
        for magnitude_idx in range(10):
            d_matrix = torch.cat(self.magnitude_to_diff[magnitude_idx])
            self.magnitude_to_diff[magnitude_idx] = d_matrix

    def compute_alignments(self) -> Dict[int, Tensor]:
        """Computes relative equivariance the baseline.
        cos_dist(b_i, b_j) - cos_dist(d_i, d_j)
        D is diff matrix: f(X) - f(T(X))
        B is baseline matrix (shuffled rows per col)
        """
        alignments = dict()
        for magnitude, d_matrix in self.magnitude_to_diff.items():
            perms = torch.randperm(d_matrix.shape[0])
            d_matrix_shuffled_rows = d_matrix[perms, :]
            # baseline
            b_matrix = self.magnitude_to_shuffled_diff[magnitude]
            b_matrix_shuffled_rows = b_matrix[perms, :]
            b_alignments = compute_cos_distance(b_matrix, b_matrix_shuffled_rows)
            d_alignments = compute_cos_distance(d_matrix, d_matrix_shuffled_rows)
            alignments[magnitude] = b_alignments - d_alignments
        return alignments

    def create_magnitude_to_shuffled_diff(self) -> Dict[int, Tensor]:
        """Creates a shuffled version of d_matrix for baselines"""
        magnitude_to_shuffled_diff = dict()
        for magnitude, d_matrix in self.magnitude_to_diff.items():
            d_matrix_shuffled = self.shuffle_d_matrix(d_matrix)
            magnitude_to_shuffled_diff[magnitude] = d_matrix_shuffled
        return magnitude_to_shuffled_diff

    @staticmethod
    def shuffle_d_matrix(d_matrix: Tensor) -> Tensor:
        """Shuffles rows within each column for baseline"""
        d_shuffled = torch.clone(d_matrix)
        for i in range(d_matrix.shape[1]):
            n = d_matrix.shape[0]
            rows_perm = torch.randperm(n)
            d_shuffled[:, i] = d_matrix[rows_perm, i]
        return d_shuffled

    def compute(self):
        """Computes baseline and regular alignments for embeddings"""
        self.create_d_matrix()
        self.magnitude_to_shuffled_diff = self.create_magnitude_to_shuffled_diff()
        self.alignments = self.compute_alignments()

    def create_digests(self):
        self.alignments_digest = TransformAlignmentsDigest(
            self.transform_name, self.data_stage
        )
        self.alignments_digest.batch_update(self.alignments)

    def on_test_end(self):
        self.compute()
        self.create_digests()
        return super().on_test_end()


class AlignmentsDigest:
    def __init__(self, data_stage: str, prefix=""):
        self.data_stage = data_stage
        self.transform_to_alignments_digest = dict()
        self.prefix = prefix

    def get_percentile(self, transform: str, magnitude: str, percentile: int) -> float:
        return (
            self.transform_to_alignments_digest[transform]
            .magnitude_to_digest[magnitude]
            .percentile(percentile)
        )

    @property
    def transform_names(self) -> List[str]:
        return list(self.transform_to_alignments_digest.keys())

    @property
    def magnitudes(self) -> List[str]:
        return [str(i) for i in range(10)]

    def update(self, model: EmbeddingAlignmentModule):
        """Extracts digests from each model"""
        self.transform_to_alignments_digest[
            model.transform_name
        ] = model.alignments_digest

    def save(self, results_dir: str):
        file_name = f"{self.prefix}alignments_{self.data_stage}.json"
        self.save_alignments(
            results_dir, file_name, self.transform_to_alignments_digest
        )

    def save_alignments(
        self, results_dir: str, file_name: str, transform_to_alignments_digest: dict
    ):
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, file_name)
        transform_to_alignments = {
            k: v.serialize() for k, v in transform_to_alignments_digest.items()
        }
        with open(path, "w") as f:
            json.dump(transform_to_alignments, f)

    def load(self, results_dir: str):
        file_name = f"{self.prefix}alignments_{self.data_stage}.json"
        self.transform_to_alignments_digest = self.load_alignments(
            results_dir, file_name
        )

    def load_alignments(
        self, results_dir: str, file_name: str
    ) -> Dict[str, TransformAlignmentsDigest]:
        path = os.path.join(results_dir, file_name)
        with open(path, "r") as f:
            alignments = json.load(f)

        transform_to_alignments_digest = dict()

        for transform in alignments:
            alignments_digest = TransformAlignmentsDigest(
                transform, data_stage=self.data_stage
            )
            alignments_digest.load_from_dict(alignments[transform])
            transform_to_alignments_digest[transform] = alignments_digest
        return transform_to_alignments_digest


class EmbeddingAlignmentViTModule(EmbeddingAlignmentModule):
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
            data_stage=data_stage,
            transform_name=transform_name,
            prefix=prefix,
        )
        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)

    def forward(self, x):
        return self.model.forward_features(x)


def compute_cos_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Return cos distance defined as 1 - cos similarity"""
    return 1 - F.cosine_similarity(x1, x2)