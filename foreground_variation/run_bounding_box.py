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
"""
import hydra
import tempfile
import shutil
from foreground_variation import bounding_boxes
from hydra.utils import instantiate
from submitit.helpers import RsyncSnapshot


@hydra.main(config_name="config_bounding_box.yaml")
def main(config):
    data_module = instantiate(config.data_module)
    bounding_boxes.main(data_module, **config.bounding_boxes)


if __name__ == "__main__":
    snapshot_dir = tempfile.mkdtemp()
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()
    shutil.rmtree(snapshot_dir)