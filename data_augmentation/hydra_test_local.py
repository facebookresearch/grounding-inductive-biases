"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import time, os, torch, hydra
import sys 
import json 
import omegaconf
sys.path.append('/real-data-transformations/')

def set_submitit_vars(args):
    """Sets rank and world_size"""
    if args.distributed:
        # job_env = submitit.JobEnvironment()
        args.rank = 0
        args.world_size = 1
    return args

@hydra.main(config_name="config_test_local.yaml")
def experiment(cfg):
    import test
    try:
        cfg.classifier.gpu
    except:
        cfg.classifier.gpu = None
    try:
        cfg.classifier.seed
    except:
        cfg.classifier.seed = None

    args = cfg.classifier
    
    checkpoint_path = os.path.join(cfg.classifier.result_dir, 'checkpoints')
    if os.path.exists(checkpoint_path):
        checkpoint_folders = os.listdir(checkpoint_path)
    else:
        checkpoint_folders = []
    
    args = set_submitit_vars(args)

    job_id = cfg.classifier.job_id
    checkpoint_folder = []
    for f in checkpoint_folders:
        f_id = ('_').join(f.split('_')[4:])
        if job_id == f_id:
            checkpoint_folder.append(f)
    if len(checkpoint_folder) > 1:
        raise Exception("More than one matching folder found.")
    elif len(checkpoint_folder)==0:
        raise Exception("No matching folder found.")
    else:
        checkpoint_folder = checkpoint_folder[0]
        print("One folder found from {}".format(checkpoint_folder))
        ckpt_path = os.path.join(checkpoint_path,'{0}/model_best.pth'.format(checkpoint_folder))
        hparams_path =  os.path.join(checkpoint_path,'{0}/hparams.json'.format(checkpoint_folder))
        with open(hparams_path, "rb") as f:
            hparams = json.load(f)
        parameters = omegaconf.dictconfig.DictConfig(hparams)
        parameters.rank = args.rank
        parameters.world_size = args.world_size
        parameters.augment_valid = args.augment_valid 
        parameters.augment_train = args.augment_train 
        parameters.no_aug_test = args.no_aug_test
        parameters.test_seeds = args.test_seeds
        parameters.scale_mag =args.scale_mag
        ####### IMPORTANT

        if os.path.exists(ckpt_path):
            print("It has a checkpoint.pth, testing")
            test.run_testing(parameters, ckpt_path)
        else:
            raise Exception("No model_best.pth found")

if __name__ == "__main__":
    experiment()
