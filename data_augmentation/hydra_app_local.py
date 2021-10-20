import time, os, torch, hydra, submitit
import pathlib
import sys 
sys.path.append('/real-data-transformations/')

def set_submitit_vars(args):
    """Sets rank and world_size"""
    if args.distributed:
        # job_env = submitit.JobEnvironment()
        args.rank = 0
        args.world_size = 1
    return args

@hydra.main(config_name="config_test_local_ddp.yaml")
def experiment(cfg):
    import data_augmentation.my_training as my_training 
    try:
        cfg.classifier.gpu
    except:
        cfg.classifier.gpu = None
    try:
        cfg.classifier.seed
    except:
        cfg.classifier.seed = None

    args = cfg.classifier
    checkpoint_path = os.path.join(args.result_dir, 'checkpoints')
    if os.path.exists(checkpoint_path):
        checkpoint_folders = os.listdir(checkpoint_path)
    else:
        checkpoint_folders = []
    job_id = ""

    args = set_submitit_vars(args)

    checkpoint_folder = []
    for f in checkpoint_folders:
        f_id = ('_').join(f.split('_')[4:])
        if job_id == f_id:
            checkpoint_folder.append(f)
    if len(checkpoint_folder) > 1:
        print(checkpoint_folder)
        raise Exception("More than one matching folder found.")
    elif len(checkpoint_folder)==0:
        print("No matching folder found.")
        my_training.run_model(cfg.classifier, ckpt_path=None, repo=None)
    else:
        checkpoint_folder = checkpoint_folder[0]
        print("One folder found from {}".format(checkpoint_folder))
        ckpt_path = os.path.join(checkpoint_path,'{0}/checkpoint.pth'.format(checkpoint_folder))
        repo_path=os.path.join(checkpoint_path,checkpoint_folder)
        if os.path.exists(ckpt_path):
            print("It has a checkpoint.pth, resuming from there")
            my_training.run_model(cfg.classifier, ckpt_path, repo=repo_path)
        else:
            print("But has no checkpoint.pth")
            my_training.run_model(cfg.classifier, ckpt_path=None, repo=repo_path)

if __name__ == "__main__":
    experiment()
