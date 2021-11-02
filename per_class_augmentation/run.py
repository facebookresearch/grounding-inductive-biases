"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Local GPU:
python run.py -m +mode=local_gpu

Cluster multi-GPU:
python run.py -m +mode=cluster

Per class augmentation runs:
python run.py -m +mode=cluster random_seed=0,3,10
python run.py -m +mode=cluster random_seed=0,3,10 data_module.num_transforms=1

Per class augmentation plus standard:
python run.py -m +mode=cluster random_seed=0,3,10 \
    data_module.similarity_type=resnet18 data_module.plus_standard_aug=True

Standard ResNet runs:
python run.py -m --config-name config_standard_aug.yaml +mode=cluster

Overfit on a 1% of data:
python run.py -m --config-name config_standard_aug.yaml +mode=local_gpu \
     +trainer.overfit_batches=0.01
"""
import pytorch_lightning as pl
import hydra
import tempfile
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from submitit.helpers import RsyncSnapshot
from per_class_augmentation.model import best_val_checkpoint_callback
from pytorch_lightning import loggers as pl_loggers


@hydra.main(config_name="config.yaml", config_path="config")
def main(config: DictConfig):
    pl.seed_everything(config.random_seed)
    loggers = create_loggers(config)
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=[best_val_checkpoint_callback],
        logger=loggers,
    )
    # per GPU
    batch_size = int(config.total_batch_size / config.trainer.gpus)
    model = instantiate(config.module)
    data_module = instantiate(config.data_module, batch_size=batch_size)
    trainer.fit(model, data_module)
    trainer.test(model)


def create_loggers(config: DictConfig) -> list:
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    tag = config.tag if config.tag else " "
    wandb_logger = pl_loggers.WandbLogger(
        project="real-data-transformations",
        name=config.name,
        # save configs with resolved values
        config=yaml.load(OmegaConf.to_yaml(config, resolve=True)),
        tags=[tag],
    )
    return [tb_logger, wandb_logger]


if __name__ == "__main__":
    snapshot_dir = tempfile.mkdtemp()
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()