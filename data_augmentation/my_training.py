"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

''' Modification of the pytorch example in https://github.com/pytorch/examples/tree/master/imagenet.'''

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.distributed as dist

print(torch.__version__)
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import data_utils
from data_utils import functions
from utils import checkpointing
from models.arch import modified_resnet18

# for augerino
import augerino_lib

print(augerino_lib.__path__)
from augerino_lib.uniform_aug import (
    MyUniformAug,
    AugModuleMin,
    UniformAugEachPos,
    UniformAugEachMin,
)
from augerino_lib.aug_modules import AugAveragedModel
from augerino_lib import losses as aug_losses
import omegaconf
import experiment_utils.experiment_repo as repository
import copy
import datetime
import json
import pickle

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def run_model(params, ckpt_path=None, repo=None):
    args = params

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        torch.cuda.set_device(args.rank)
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method="tcp://{}:{}".format("localhost", 10001),
            world_size=args.world_size,
            rank=args.rank,
        )
    main_worker(args.gpu, ngpus_per_node, args, ckpt_path, repo)

    # cleanup distributed
    if args.distributed:
        cleanup_distributed()


def cleanup_distributed():
    dist.destroy_process_group()


def main_worker(gpu, ngpus_per_node, args, ckpt_path, repo):
    global best_acc1
    args.gpu = gpu

    if not args.distributed or (args.distributed and args.rank % ngpus_per_node == 0):
        model_dir = create_repo(args, repo)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    cur_device = torch.cuda.current_device()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        net = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == "modified_resnet18":
            net = modified_resnet18(modify=args.modify, num_classes=args.num_classes)
        else:
            net = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.augerino:
        if args.inv_per_class:
            assert args.disable_at_valid
            augerino_classes = args.num_classes
        else:
            augerino_classes = 1
        if args.transfos == ["tx", "ty", "scale"]:  # special case we pass it 1 by 1
            if args.min_val:
                print("Using UniformAugEachMin")
                augerino = UniformAugEachMin(
                    transfos=args.transfos,
                    min_values=args.min_values,
                    shutvals=args.shutdown_vals,
                    num_classes=augerino_classes,
                )
            else:
                print("Using UniformAugEach")
                augerino = UniformAugEachPos(
                    transfos=args.transfos,
                    shutvals=args.shutdown_vals,
                    num_classes=augerino_classes,
                )
        else:
            if args.min_val:
                augerino = AugModuleMin(
                    transfos=args.transfos,
                    min_values=args.min_values,
                    shutvals=args.shutdown_vals,
                    num_classes=augerino_classes,
                )
            else:
                augerino = MyUniformAug(
                    transfos=args.transfos,
                    shutvals=args.shutdown_vals,
                    num_classes=augerino_classes,
                )

        augerino.set_width(
            torch.FloatTensor(args.startwidth)[None,:].repeat(augerino_classes,1)
        )
        print(augerino.width)
        if args.fixed_augerino:
            augerino.width.requires_grad=False
        model = AugAveragedModel(
            net, augerino, disabled=False, ncopies=args.ncopies, onecopy=args.onecopy
        )
    else:
        model = net

    # save initial width
    if args.augerino:
        widths = [model.aug.width.clone().detach()]
    else:
        widths = []

    model = model.cuda(device=cur_device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    if (
        args.pretrained_noaugment and ckpt_path is None
    ):  # to ensure we're not restarting after preemption
        if args.distributed:
            net = torch.nn.parallel.DistributedDataParallel(
                module=net, device_ids=[cur_device], output_device=cur_device
            )
        checkpointing.restart_from_checkpoint(
            args.noaugment_path, args, state_dict=net
        )  # WARNING: Lr is not adjusted accordingly
        factor_lr = 1.0
    else:
        factor_lr = 0.1

    # define loss function (criterion) and optimizer
    if args.augerino:
        criterion = aug_losses.safe_unif_aug_loss_each
        model_param = (
            model.module.model.parameters()
            if args.distributed
            else model.model.parameters()
        )
        aug_param = (
            model.module.aug.parameters()
            if args.distributed
            else model.aug.parameters()
        )
        params = [
            {
                "name": "model",
                "params": model_param,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
            },
            {
                "name": "aug",
                "params": aug_param,
                "momentum": args.momentum,
                "weight_decay": 0.0,
                "lr": args.lr * factor_lr,
            },
        ]
    else:
        criterion = nn.CrossEntropyLoss().cuda()
        params = [
            {
                "name": "model",
                "params": model.parameters(),
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
            }
        ]

    optimizer = torch.optim.SGD(params, args.lr)
    to_restore = {"epoch": 0, "best_acc1": 0.0, "all_acc1": [], "width": widths}
    if ckpt_path is not None:
        checkpointing.restart_from_checkpoint(
            ckpt_path,
            args,
            run_variables=to_restore,
            state_dict=model,
            optimizer=optimizer,
        )
    args.start_epoch = to_restore["epoch"]
    best_acc1 = to_restore["best_acc1"]
    all_acc1 = to_restore["all_acc1"]
    widths = to_restore["width"]
    print("Starting from Epoch", args.start_epoch)

    cudnn.benchmark = True
    # Data loading code
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    train_loader, val_loader, train_sampler = functions.return_loader_and_sampler(
        args, traindir, valdir
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        # return

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args, factor_lr)

        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5, val_loss = validate(val_loader, model, criterion, args)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        all_acc1.append(acc1)
        if args.augerino:
            width = model.module.aug.width if args.distributed else module.aug.width
            widths.append(width.clone().detach())

        if not args.distributed or (
            args.distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "acc1": acc1,
                    "acc5": acc5,
                    "all_acc1": all_acc1,
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "width": widths,
                },
                is_best,
                model_dir,
            )


def create_repo(args, repo):
    if repo is None:
        # Initialize the Repo
        print("Creating repo..")
        exp_repo = repository.ExperimentRepo(
            local_dir_name="json_files", root_dir=args.result_dir
        )

        # Create new experiment
        cfg_copy = copy.deepcopy(dict(args))
        for i in cfg_copy.keys():
            if type(cfg_copy[i]) == omegaconf.listconfig.ListConfig:
                cfg_copy[i] = list(cfg_copy[i])
                # in case of nested list
                for j, el in enumerate(cfg_copy[i]):
                    if type(el) == omegaconf.listconfig.ListConfig:
                        cfg_copy[i][j] = list(el)
        exp_id = exp_repo.create_new_experiment(cfg_copy)
        cfg_copy["id"] = exp_id

        # Set up model directory
        current_time = datetime.datetime.now().strftime(r"%y%m%d_%H%M")
        ckpt_dir = os.path.join(args.result_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        model_dir = os.path.join(ckpt_dir, "ckpt_{}_{}".format(current_time, exp_id))

        # Save hyperparameter settings
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(os.path.join(model_dir, "hparams.json")):
            with open(os.path.join(model_dir, "hparams.json"), "w") as f:
                json.dump(cfg_copy, f, indent=2, sort_keys=True)
            with open(os.path.join(model_dir, "hparams.pkl"), "wb") as f:
                pickle.dump(cfg_copy, f)
    else:
        model_dir = repo
    return model_dir


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.4f")
    top5 = AverageMeter("Acc@5", ":6.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    epoch_start = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        if args.augerino and args.inv_per_class:
            output = model(images, target)
        else:
            output = model(images)
        if args.augerino:
            loss = criterion(output, target, model, args, reg=args.aug_reg)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        n_points = torch.FloatTensor([images.size(0)])
        if torch.cuda.is_available():
            n_points = n_points.cuda(non_blocking=True)
        
        if args.distributed:
            torch.distributed.all_reduce(acc1)
            torch.distributed.all_reduce(acc5)
            torch.distributed.all_reduce(n_points)
        
        acc1 = acc1 / n_points * 100.0
        acc5 = acc5 / n_points * 100.0

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], n_points.item())
        top5.update(acc5[0], n_points.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    epoch_end = time.time()
    print("time 1 epoch {:.3f}".format(epoch_end - epoch_start))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.4f")
    top5 = AverageMeter("Acc@5", ":6.4f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    if args.augerino and args.disable_at_valid:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.disabled = True
        elif isinstance(model, AugAveragedModel):
            model.disabled = True
        print("Disabling Augerino")

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda(non_blocking=True)

            # compute output
            if args.augerino and args.inv_per_class:
                output = model(images, target)
            else:
                output = model(images)
            if args.augerino:
                loss = criterion(output, target, model, args, reg=args.aug_reg)
            else:
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            acc1 = acc1 / float(images.size(0)) * 100.0
            acc5 = acc5 / float(images.size(0)) * 100.0

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )
    if args.augerino and args.disable_at_valid:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.disabled = False
        elif isinstance(model, AugAveragedModel):
            model.disabled = False
    return top1.avg.item(), top5.avg.item(), losses.avg


def save_checkpoint(state, is_best, directory):
    filename = "{0}/checkpoint.pth".format(directory)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "{0}/model_best.pth".format(directory))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args, lr_factor):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        if param_group["name"] != "aug":
            param_group["lr"] = lr
        else:
            param_group["lr"] = lr * lr_factor


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res


if __name__ == "__main__":
    import sys

    run_model(sys.argv[1:])