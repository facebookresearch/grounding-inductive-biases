"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging, time, os, torch, hydra
from pathlib import Path
import submitit

log = logging.getLogger(__name__)


@hydra.main(config_name="conf")
def launch(cfg):
    executor = submitit.AutoExecutor(folder=cfg.logging_folder, slurm_max_num_timeout=3)
    executor.update_parameters(
        timeout_min=1,
        slurm_partition="dev",
        gpus_per_node=0,
        cpus_per_task=1,
        slurm_signal_delay_s=2,
    )
    training_callable = ModelTrainer()
    job = executor.submit(training_callable, cfg.checkpoint_path)
    time.sleep(5)
    # Fake preemption for testing
    job._interrupt(timeout=False)


class ModelTrainer:
    def __init__(self):
        # this is the "state" which we will be able to access when checkpointing:
        self.model = 0

    def __call__(self, checkpointpath: str):
        if Path(checkpointpath).exists():
            self.model = torch.load(checkpointpath)

        for i in range(self.model, 20):
            self.model = i
            time.sleep(2)
            print(f"Hello world {i}")

    def checkpoint(self, checkpointpath: str) -> submitit.helpers.DelayedSubmission:
        # the checkpoint method is called asynchroneously when the slurm manager
        # sends a preemption signal, with the same arguments as the __call__ method
        # "self" is your callable, at its current state.
        # "self" therefore holds the current version of the model:
        # do whatever you need to do to dump it properly
        torch.save(self.model + 1, checkpointpath)
        # create a new, clean (= no loaded model) NetworkTraining instance which
        # will be loaded when the job resumes, and will fetch the dumped model
        # (creating a new instance is not necessary but can avoid weird interactions
        # with the current instance)
        training_callable = ModelTrainer()
        # Resubmission to the queue is performed through the DelayedSubmission object
        return submitit.helpers.DelayedSubmission(training_callable, checkpointpath)


if __name__ == "__main__":
    launch()