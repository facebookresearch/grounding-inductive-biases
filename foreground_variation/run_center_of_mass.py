"""
Local GPU:
python run.py -m +mode=local_gpu

Cluster multi-GPU:
python run.py -m +mode=cluster
"""
import hydra
import tempfile
import shutil
from foreground_variation import center_of_mass
from hydra.utils import instantiate
from submitit.helpers import RsyncSnapshot


@hydra.main(config_name="config_center_of_mass.yaml")
def main(config):
    data_module = instantiate(config.data_module)
    center_of_mass.main(data_module, **config.center_of_mass)


if __name__ == "__main__":
    snapshot_dir = tempfile.mkdtemp()
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()
    shutil.rmtree(snapshot_dir)