import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.mytransforms import (
    RandomSizeResizedCenterCrop,
    RandomSizeResizedCenterCropSquaredUniformInv, 
    RandomResizedCropExponential,
    RandomResizedCropBeta,
    )
import numbers


class CircularPad:
    def __init__(self, padding_mode="circular", fill=0):
        assert isinstance(fill, (numbers.Number, str, tuple))

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        _, h, w = img.size()
        padded_img = torch.nn.functional.pad(
            img.unsqueeze(0), pad=(w, w, h, h), mode=self.padding_mode, value=self.fill
        )
        return padded_img.squeeze(0)


class MyCenterCrop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        _, h, w = img.size()
        size = (round(h / 3), round(w / 3))
        return transforms.functional.center_crop(img, size)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


def return_loader_and_sampler(args, traindir, valdir, return_train = True):

    if args.tvalues is None:
        args.tvalues = [1,1]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if return_train:
        if args.augment == "HC":  # flip + random size and location crop
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            224
                        ),  # resize is not needed since the crop will be in function of the size and ratio of the image
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "H":  # flip only
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "C":  # random size and location crop only
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            224
                        ),  # resize is not needed since the crop will be in function of the size and ratio of the image
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "CC":  # random size center crop only
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        RandomSizeResizedCenterCrop(224, interpolation=interpolation),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "TS":  # Translation + randomResizedCrop
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.ToTensor(),  # torch.nn.functional.pad needs a tensor as input
                        transforms.RandomAffine(
                            0,
                            translate=(args.tvalues[0], args.tvalues[1]),
                            scale=None,
                            shear=None,
                            fill=0,
                            interpolation=interpolation,
                        ),  # translate
                        RandomSizeResizedCenterCrop(224, interpolation=interpolation),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "TSnoAR":  # Translation + randomResizedCrop no aspect ratio
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.ToTensor(),  # torch.nn.functional.pad needs a tensor as input
                        transforms.RandomAffine(
                            0,
                            translate=(args.tvalues[0], args.tvalues[1]),
                            scale=None,
                            shear=None,
                            fill=0,
                            interpolation=interpolation,
                        ),  # translate
                        RandomSizeResizedCenterCrop(224, ratio = (1.,1.),interpolation=interpolation),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "CCTSnoAR": # CenterCrop + TSnoAR
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.Resize(256),  # ensures that the minimal size is 256
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),  # torch.nn.functional.pad needs a tensor as input
                        transforms.RandomAffine(
                            0,
                            translate=(args.tvalues[0], args.tvalues[1]),
                            scale=None,
                            shear=None,
                            fill=0,
                            interpolation=interpolation,
                        ),  # translate
                        RandomSizeResizedCenterCrop(224, ratio = (1.,1.),interpolation=interpolation),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "TSnoARSQRTInv": # CenterCrop + TSnoAR with heavy tail
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.Resize(256),  # ensures that the minimal size is 256
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),  # torch.nn.functional.pad needs a tensor as input
                        transforms.RandomAffine(
                            0,
                            translate=(args.tvalues[0], args.tvalues[1]),
                            scale=None,
                            shear=None,
                            fill=0,
                            interpolation=interpolation,
                        ),  # translate
                        RandomSizeResizedCenterCropSquaredUniformInv(224, ratio = (1.,1.),interpolation=interpolation),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "noCCTSnoARSQRTInv": # TSnoAR with heavy tail
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.ToTensor(),  # torch.nn.functional.pad needs a tensor as input
                        transforms.RandomAffine(
                            0,
                            translate=(args.tvalues[0], args.tvalues[1]),
                            scale=None,
                            shear=None,
                            fill=0,
                            interpolation=interpolation,
                        ),  # translate
                        RandomSizeResizedCenterCropSquaredUniformInv(224, ratio = (1.,1.),interpolation=interpolation),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "R":  # Random Crop, fixed size
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.Resize(256),  # ensures that the minimal size is 256
                        transforms.RandomCrop(args.cropsize),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "None":  # center crop, fixed size
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.Resize(256),  # ensures that the minimal size is 256
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "Cexp": 
            print("Using Exponential") 
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        RandomResizedCropExponential(
                            224, scale=(args.scale_exp,),
                            interpolation=interpolation
                        ),  # resize is not needed since the crop will be in function of the size and ratio of the image
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "Cbeta": 
            print("Using Beta") 
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        RandomResizedCropBeta(
                            224, scale=(1.0, args.scale_exp),
                            interpolation=interpolation
                        ),  # resize is not needed since the crop will be in function of the size and ratio of the image
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "T":  # Translation only
            print("Using T")
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.Resize(256),  
                        transforms.ToTensor(),  
                        transforms.RandomAffine(
                            0,
                            translate=(args.tvalues[0], args.tvalues[1]),
                            scale=None,
                            shear=None,
                            fill=0,
                            interpolation=interpolation,
                        ),  # translate
                        transforms.CenterCrop(224),
                        normalize,
                    ]
                ),
            )
        elif args.augment == "T2": 
            print("Using T2")
            interpolation = transforms.RandomResizedCrop(224).interpolation
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.Resize(256),  # ensures that the minimal size is 256
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),  # torch.nn.functional.pad needs a tensor as input
                        transforms.RandomAffine(
                            0,
                            translate=(args.tvalues[0], args.tvalues[1]),
                            scale=None,
                            shear=None,
                            fill=0,
                            interpolation=interpolation,
                        ),  # translate
                        normalize,
                    ]
                ),
            )
        print(args.workers)
    else:
        train_dataset = []

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # per GPU for DistributedDataParallel
        batch_size = int(args.batch_size / args.world_size)
        print(f"batch size per GPU is {batch_size}")
    else:
        train_sampler = None
        batch_size = args.batch_size

    if return_train:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
    else:
        train_loader = None
    print("Train loader initiated")
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    print("Val loader initiated")
    return train_loader, val_loader, train_sampler
