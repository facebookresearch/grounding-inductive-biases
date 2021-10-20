"""
Local GPU Test:
python run.py -m +mode=local_gpu train_proportion_to_sample=0.0001

Cluster multi-GPU:
python run.py -m +mode=cluster
"""
import pytorch_lightning as pl
import hydra
import tempfile
from hydra.utils import instantiate
from submitit.helpers import RsyncSnapshot
from equivariance_measure import embedding_distances, transformations


@hydra.main(config_name="config_invariance.yaml", config_path="config")
def main(config):
    pl.seed_everything(config.random_seed)
    trainer = pl.Trainer(**config.trainer, limit_test_batches=config.n_batches)
    data_module = instantiate(config.data_module)
    dataloader = (
        data_module.train_dataloader()
        if config.data_stage == "train"
        else data_module.val_dataloader()
    )
    run_across_transforms(config.module, dataloader, trainer, config.data_stage)


def run_across_transforms(module, dataloader, trainer, data_stage):
    transforms = transformations.TRANSFORMATION_NAMES
    cos_distance_invariance_digest = embedding_distances.InvariancesDigest(
        "cos_distance",
        data_stage,
        prefix=module.prefix,
    )
    l2_invariance_digest = embedding_distances.InvariancesDigest(
        "l2", data_stage, prefix=module.prefix
    )

    for transform in transforms:
        model = instantiate(module, transform_name=transform)
        trainer.test(model=model, test_dataloaders=dataloader)
        cos_distance_invariance_digest.update(model)
        l2_invariance_digest.update(model)

    if model.results_dir:
        cos_distance_invariance_digest.save(model.results_dir)
        l2_invariance_digest.save(model.results_dir)


if __name__ == "__main__":
    snapshot_dir = tempfile.mkdtemp()
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()