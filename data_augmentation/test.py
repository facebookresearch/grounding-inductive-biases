import os
import random
import warnings

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.models as models
from data_utils import functions_bis
from utils import checkpointing
from augerino_lib.uniform_aug import (
    MyUniformAug,
    AugModuleMin,
    UniformAugEachPos,
    UniformAugEachMin,
)
from augerino_lib.aug_modules import AugAveragedModel
from data_augmentation.my_training import (
    accuracy, 
    AverageMeter, 
    cleanup_distributed,
)

def run_testing(params, ckpt_path=None):
    args=params

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        torch.cuda.set_device(args.rank)
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method="tcp://{}:{}".format("localhost", 10001),
            world_size=args.world_size,
            rank=args.rank,
        )

    main_worker(args.gpu, ngpus_per_node, args, ckpt_path)
    # cleanup distributed
    if args.distributed:
        cleanup_distributed()

def main_worker(gpu, ngpus_per_node, args, ckpt_path):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))

    cur_device = torch.cuda.current_device()
    # create model
    print("=> creating model '{}'".format(args.arch))
    net = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.augerino:
        if args.inv_per_class:
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
        model = AugAveragedModel(net, augerino, ncopies=args.ncopies)
    else:
        model = net

    model = model.cuda(device=cur_device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )

    to_restore = {"epoch":0.0, "all_acc1": 0.0, "best_acc1": 0.0}
    checkpointing.restart_from_checkpoint(ckpt_path, args, 
                    run_variables=to_restore,state_dict=model)

    best_acc1 = to_restore["best_acc1"]
    model_acc1 = to_restore["all_acc1"][-1]
    print("Best Acc1 was", best_acc1)
    print("Model Acc1 was", model_acc1)

    # Data loading code
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    _, test_loader, _ = functions_bis.return_loader_and_sampler(args, traindir, valdir, return_train=False) 

    compute_per_sample = True #TODO: better than manually set 
    if compute_per_sample:
        assert len(args.test_seeds)==1
    top1s = []
    top5s = []
    tops1class = []
    per_img_dict = {}
    for seed in args.test_seeds:
        # Set seed for testing 
        random.seed(seed)
        torch.manual_seed(seed)

        top1, top5, top1class = test(test_loader, model, args, per_img_dict, args.no_aug_test, compute_per_sample)
        print('{:.5f},{:.5f},{:.5f}'.format(top1, top5, top1class.mean()))
        top1s.append(torch.FloatTensor([top1]))
        top5s.append(torch.FloatTensor([top5]))
        tops1class.append(top1class[None,:])
    print(top1s)
    top1s = torch.cat(top1s).mean()
    top5s = torch.cat(top5s).mean()
    tops1class = torch.cat(tops1class,dim=0).mean(0)

    ckpt = {}
    ckpt['top1'] = top1s
    ckpt['top5'] = top5s
    ckpt['top1class'] = tops1class
    name = 'test3_{0}_{1}_{2}.pth'.format(args.augment_valid, args.scale_mag, args.no_aug_test)
    ckpt_folder = '/'.join(ckpt_path.split('/')[:-1])
    if not args.distributed or (
            args.distributed and args.rank % ngpus_per_node == 0
        ):
        torch.save(ckpt, os.path.join(ckpt_folder, name))
        if compute_per_sample:
            name = 'per_sample_{0}_{1}_{2}.pth'.format(args.augment_valid, args.scale_mag, args.no_aug_test)
            torch.save(per_img_dict, os.path.join(ckpt_folder, name))

def test(loader, model, args, per_img_dict, disable_augerino=True, compute_per_sample=False):
    # switch to evaluate mode
    model.eval()

    losses = []
    top1 = AverageMeter("Acc@1", ":6.4f")
    top5 = AverageMeter("Acc@5", ":6.4f")
    count = 0
    acc_per_class = torch.zeros(args.num_classes)
    count_per_class = torch.zeros(args.num_classes)
    if args.augerino:
        if disable_augerino:
            if isinstance(model, nn.parallel.DistributedDataParallel):
                model.module.disabled = True
            elif isinstance(model, AugAveragedModel):
                model.disabled = True
            print("Disabling Augerino")
        else:
            if isinstance(model, nn.parallel.DistributedDataParallel):
                model.module.disabled = False
            elif isinstance(model, AugAveragedModel):
                model.disabled = False
            print("Enabling Augerino")

    with torch.no_grad():
        for i, (images, target, pathes) in enumerate(loader):
            bs = images.size(0)
            count+= bs
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            if args.augerino and args.inv_per_class:
                output = model(images,target)
            else:
                output = model(images)

            # measure accuracy 
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if compute_per_sample:
                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                correct=correct.float().squeeze(0)
                for j in range(bs):
                    per_img_dict[pathes[j]] = correct[j]

            acc1_class, count_class = acc1_per_class(args, output, target)

            acc1 = acc1 / float(bs) * 100.0
            acc5 = acc5 / float(bs) * 100.0

            top1.update(acc1[0], bs)
            top5.update(acc5[0], bs)

            acc_per_class += acc1_class * 100.0
            count_per_class += count_class

    return top1.avg.item(), top5.avg.item(), acc_per_class/count_per_class

def acc1_per_class(args, output, target):
    """Computes the per class accuracy over the k top predictions for the specified values of k"""
    acc_per_class = torch.zeros(args.num_classes)
    count_per_class = torch.zeros(args.num_classes)

    with torch.no_grad():

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct=correct.float().squeeze(0)
        classes_batch = target.unique()
        for c in classes_batch:
            acc_per_class[c] = correct[target==c].sum(0) 
            count_per_class[c] = (target==c).float().sum()
        return acc_per_class, count_per_class