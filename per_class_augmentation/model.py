"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchvision.models import resnet18
from torchmetrics import Metric
from torch.nn import functional as F
from collections import defaultdict
from typing import Dict


class ResNet18Classifier(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        momentum=0.9,
        weight_decay=0.01,
        num_classes=1000,
        pretrained=False,
        track_per_class_accruacy=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.track_per_class_accruacy = track_per_class_accruacy

        self.model = resnet18(num_classes=num_classes, pretrained=pretrained)

        self.val_top1_accuracy = pl.metrics.Accuracy()
        self.test_top1_accuracy = pl.metrics.Accuracy()
        self.val_top5_accuracy = pl.metrics.Accuracy(top_k=5)
        self.test_top5_accuracy = pl.metrics.Accuracy(top_k=5)

        self.val_per_class_accuracy = pl.metrics.Accuracy(
            average="macro", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        out = self(x)
        loss = F.cross_entropy(out, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, step="val"):
        x, labels = batch
        out = self.model(x)
        loss = F.cross_entropy(out, labels)
        preds = F.softmax(out, dim=1)
        self.log(f"{step}_loss", loss)
        self.log_accuracy(preds, labels, step=step)
        if self.track_per_class_accruacy:
            self.log(
                f"{step}_per_class_accuracy", self.val_per_class_accuracy(preds, labels)
            )
        return loss

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, step="test")

    def log_accuracy(self, preds, labels, step="train", on_epoch=True):
        """Logs top 1 and top 5 accuracy"""
        self.log(
            f"{step}_top_1_acc",
            getattr(self, f"{step}_top1_accuracy")(preds, labels),
            prog_bar=True,
            on_epoch=on_epoch,
        )
        self.log(
            f"{step}_top_5_acc",
            getattr(self, f"{step}_top5_accuracy")(preds, labels),
            on_epoch=on_epoch,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]


best_val_checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    filename="best-val-{epoch:02d}-{val_loss:.2f}-{val_top_1_acc:.2f}",
    save_top_k=1,
    mode="min",
)


class PerClassAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False, num_classes=1000):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "correct", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        target_unique, target_counts = target.unique(return_counts=True)
        correct_indices, correct_counts = target[preds == target].unique(
            return_counts=True
        )

        self.total[target_unique] += target_counts
        self.correct[correct_indices] += correct_counts

    def compute(self):
        return self.correct.float() / self.total
