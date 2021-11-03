"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Locally:
python run.py -m launch_env=local hydra/launcher=submitit_local similarity_search.run_on_sample_classes=True \
    similarity_search.max_n_image_pairs=10 similarity_search.batch_size=4

Locally parallelize classes:
python run.py -m launch_env=local hydra/launcher=submitit_local similarity_search.run_on_sample_classes=True \
    similarity_search.max_n_image_pairs=20 similarity_search.batch_size=4 \
    similarity_search.parallel_classes.parallelize=True \
    similarity_search.parallel_classes.partition_index='range(0,10)' 


Cluster:
python run.py -m

Cluster parallelize classes:
python run.py -m similarity_search.parallel_classes.parallelize=True \
      similarity_search.parallel_classes.num_partitions=100 \
      similarity_search.parallel_classes.partition_index='range(0,100)' \
      similarity_search.similarity_type=resnet18_no_aug 
"""

import hydra
import logging
import os
from pathlib import Path
from omegaconf import DictConfig
from similarity_search_experiments import similarity_search


log = logging.getLogger(__name__)


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    output_dir = Path(os.getcwd())

    log.info(f"{cfg=}")

    class_labels = None
    if cfg.similarity_search.run_on_sample_classes:
        if cfg.launch_env.name == "local":
            class_labels = cfg.similarity_search.sample_train_class_labels
        elif cfg.launch_env.name == "cluster":
            class_labels = cfg.similarity_search.sample_val_class_labels

        log.info(f"running on {len(class_labels)} sample classes")

    similarity_search.main(
        class_labels,
        similarity_type=cfg.similarity_search.similarity_type,
        batch_size=cfg.similarity_search.batch_size,
        max_n_image_pairs=cfg.similarity_search.max_n_image_pairs,
        use_val=cfg.similarity_search.use_val,
        parallelize_classes=cfg.similarity_search.parallel_classes.parallelize,
        num_partitions=cfg.similarity_search.parallel_classes.num_partitions,
        partition_index=cfg.similarity_search.parallel_classes.partition_index,
        run_on_subpolicies=cfg.similarity_search.run_on_subpolicies,
        save_path=output_dir,
    )


if __name__ == "__main__":
    main()