import random
import os

import numpy as np

import torch

from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

# Fix random seed
def fix_seed(seed = 0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def build_transform(is_train, args):
    if args.dataset == 'mnist':
        t = []
        size = int((256 / 224) * args.input_size)
        t.append(transforms.Resize(size, interpolation=3)) # to maintain same ratio w.r.t. 224 images
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize((0.1307,),(0.3081,)))
        return transforms.Compose(t)
    
    else:
        resize_im = args.input_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
            return transform

        t = []
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(transforms.Resize(size, interpolation=3)) # to maintain same ratio w.r.t. 224 images
            t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

