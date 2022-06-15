"""
# * Original Code
# * https://github.com/youweiliang/evit/blob/master/helpers.py
# * Modification:
# * Remove unnecessary parts for my model
"""

import time
import torch
from torchprofile import profile_macs

def speed_test(model, args, ntest=100, batchsize=64, x=None, **kwargs):
    if x is None:
        img_size = model.img_size
        x = torch.rand(batchsize, args.in_chans, *img_size).cuda()
    else:
        batchsize = x.shape[0]
    model.eval()

    start = time.time()
    for i in range(ntest):
        model(x, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    elapse = end - start
    speed = batchsize * ntest / elapse
    # speed = torch.tensor(speed, device=x.device)
    # torch.distributed.broadcast(speed, src=0, async_op=False)
    # speed = speed.item()
    return speed


def get_macs(model, args, x=None):
    model.eval()
    if x is None:
        img_size = model.img_size
        x = torch.rand(1, args.in_chans, *img_size).cuda()
    macs = profile_macs(model, x)
    return macs
