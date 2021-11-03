"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Local GPU Test:
python run.py -m +mode=local_gpu train_proportion_to_sample=0.0001

Cluster multi-GPU:
python run.py -m +mode=cluster
"""
from equivariance_measure import transformations
import pytorch_lightning as pl
import hydra
import tempfile
from hydra.utils import instantiate
from submitit.helpers import RsyncSnapshot
from equivariance_measure import embedding_alignments


@hydra.main(config_name="config_equivariance.yaml", config_path="config")
def main(config):
    pl.seed_everything(config.random_seed)
    trainer = pl.Trainer(**config.trainer, limit_test_batches=config.n_samples)
    data_module = instantiate(config.data_module)
    dataloader = (
        data_module.train_dataloader()
        if config.data_stage == "train"
        else data_module.val_dataloader()
    )
    run_across_transforms(config.module, dataloader, trainer, config.data_stage)


def run_across_transforms(module, dataloader, trainer, data_stage):
    transforms = transformations.TRANSFORMATION_NAMES
    alignments = embedding_alignments.AlignmentsDigest(data_stage, prefix=module.prefix)

    for transform in transforms:
        model = instantiate(module, transform_name=transform)
        trainer.test(model=model, test_dataloaders=dataloader)
        alignments.update(model)
    alignments.save(model.results_dir)


if __name__ == "__main__":
    snapshot_dir = tempfile.mkdtemp()
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()