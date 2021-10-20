import os
import torch
import torch.distributed as dist
import signal
import time


def trigger_job_requeue():
    """Submit a new job to resume from checkpoint.
    Be careful to use only for main process.
    """
    if (
        int(os.environ["SLURM_PROCID"]) == 0
        and str(os.getpid()) == os.environ["MAIN_PID"]
    ):
        print("time is up, back to slurm queue", flush=True)
        command = "scontrol requeue " + os.environ["SLURM_JOB_ID"]
        print(command)
        if os.system(command):
            raise RuntimeError("requeue failed")
        print("New job submitted to the queue", flush=True)
    exit(0)


def SIGTERMHandler(a, b):
    print("received sigterm")
    pass


def signalHandler(a, b):
    print("Signal received", a, time.time(), flush=True)
    os.environ["SIGNAL_RECEIVED"] = "True"
    return


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    os.environ["SIGNAL_RECEIVED"] = "False"
    os.environ["MAIN_PID"] = str(os.getpid())

    signal.signal(signal.SIGUSR1, signalHandler)
    signal.signal(signal.SIGTERM, SIGTERMHandler)
    print("Signal handler installed.", flush=True)


def restart_from_checkpoint(ckp_path, args, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    print("Found checkpoint at {}".format(ckp_path))

    if args.distributed:
        dist.barrier()
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % args.rank}
        # open checkpoint file
        checkpoint = torch.load(ckp_path, map_location=map_location)
    else:
        checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                # for compatibility with previous versions of augerino where
                # width and min_values were 1d
                if key == "state_dict":
                    if "module.aug.width" in checkpoint[key]:
                        if len(checkpoint[key]["module.aug.width"].size()) == 1:
                            print("extending the size of width")
                            checkpoint[key]["module.aug.width"] = checkpoint[key][
                                "module.aug.width"
                            ].unsqueeze(0)
                    if "module.aug.min_values" in checkpoint[key]:
                        if len(checkpoint[key]["module.aug.min_values"].size()) == 1:
                            print("extending the size of min_values")
                            checkpoint[key]["module.aug.min_values"] = checkpoint[key][
                                "module.aug.min_values"
                            ].unsqueeze(0)
                msg = value.load_state_dict(checkpoint[key], strict=True)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            print("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
