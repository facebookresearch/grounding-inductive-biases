'''Extension (with our modification) from the original Augerino code https://github.com/g-benton/learning-invariances'''

import torch


def safe_unif_aug_loss(
    outputs, labels, model, args, base_loss_fn=torch.nn.CrossEntropyLoss(), reg=0.01
):

    base_loss_fn = base_loss_fn.cuda() if args.gpu else base_loss_fn
    base_loss = base_loss_fn(outputs, labels)
    sp = torch.nn.Softplus().cuda() if args.gpu else torch.nn.Softplus()
    width = sp(model.module.aug.width) if args.distributed else sp(model.aug.width)
    aug_loss = (width).norm()
    shutdown = torch.all(width < 10)
    return base_loss - reg * aug_loss * shutdown

def safe_unif_aug_loss_each(
    outputs, labels, model, args, base_loss_fn=torch.nn.CrossEntropyLoss(), reg=0.01
):

    base_loss_fn = base_loss_fn.cuda() if args.gpu else base_loss_fn
    base_loss = base_loss_fn(outputs, labels)
    sp = torch.nn.Softplus().cuda() if args.gpu else torch.nn.Softplus()
    width = sp(model.module.aug.width) if args.distributed else sp(model.aug.width)
    shutvals = model.module.aug.shutvals if args.distributed else model.aug.shutvals
    shutdown = 1-((width - shutvals[None,:])>0).float()
    aug_loss = (shutdown*width).norm()
    return base_loss - reg * aug_loss 


def safe_unif_aug_loss_p1(
    outputs, labels, model, args, base_loss_fn=torch.nn.CrossEntropyLoss(), reg=0.01
):

    base_loss = base_loss_fn(outputs, labels).cuda(args.gpu)
    sp = torch.nn.Softplus()
    width = sp(model.aug.p1)
    aug_loss = (width).norm().cuda(args.gpu)
    shutdown = torch.all(width < 10)
    return base_loss - reg * aug_loss * shutdown


def unif_aug_loss(
    outputs, labels, model, args, base_loss_fn=torch.nn.CrossEntropyLoss(), reg=0.01
):

    base_loss_fn = base_loss_fn.cuda() if args.gpu else base_loss_fn
    base_loss = base_loss_fn(outputs, labels)

    sp = torch.nn.Softplus().cuda() if args.gpu else torch.nn.Softplus()
    width = sp(model.module.aug.width) if args.distributed else sp(model.aug.width)
    aug_loss = (width).norm()
    return base_loss - reg * aug_loss

def correct_unif_aug_loss(outputs, labels, model, 
                            args, base_loss_fn=torch.nn.CrossEntropyLoss(), 
                            reg=0.01):

    base_loss_fn = base_loss_fn.cuda() if args.gpu else base_loss_fn
    base_loss = base_loss_fn(outputs, labels)

    sp = torch.nn.Softplus().cuda() if args.gpu else torch.nn.Softplus()
    width = sp(model.module.aug.width) if args.distributed else sp(model.aug.width)
    aug_loss = (width).norm()**2 #power 2 because torch.norm() returns squared root
    shutdown = torch.all(width < 10) 
    return base_loss - reg * aug_loss * shutdown

def mlp_aug_loss(
    outputs, labels, model, base_loss_fn=torch.nn.CrossEntropyLoss(), reg=0.01
):

    base_loss = base_loss_fn(outputs, labels)
    aug_loss = model.aug.weights.norm()

    return base_loss - reg * aug_loss
