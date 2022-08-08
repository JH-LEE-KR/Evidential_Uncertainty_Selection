import time
import math
import os
import datetime
import numpy as np

import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from timm.utils import accuracy, AverageMeter
from PIL import Image
import matplotlib.pyplot as plt

from utils import *
from losses import *




def train_model(model, criterion, dataloader, optimizer, scheduler, args, logger):
    model.train()
    
    avg_epoch_time = 0

    for epoch in range(0, args.epoch):

        num_steps = len(dataloader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()

        start = time.time()
        end = time.time()

        for batch_idx, (input, target) in enumerate(dataloader):
            input, target = input.to(args.device), target.to(args.device)

            if args.uncertainty:
                one_hot = one_hot_embedding(target, args.num_classes)
                one_hot = one_hot.to(args.device)

            output, _ = model(input)

            loss = criterion(output, target)

            with torch.no_grad():
                # Measure uncertainty
                evidence = relu_evidence(output)
                alpha = evidence + 1
                uncertainty = args.num_classes / torch.sum(alpha, dim=1, keepdim=True)
                uncertainty = torch.mean(uncertainty)

            # Measure accuracy and record loss
            loss_meter.update(loss.item(), target.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # Compute gradient and update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - batch_idx)
                logger.info(
                    f'Train: Epoch[{epoch+1:{int(math.log10(args.epoch))+1}}/{args.epoch}][{batch_idx:{int(math.log10(len(dataloader)))+1}}/{num_steps}]\t'
                    f'Eta {datetime.timedelta(seconds=int(etas))}\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Avg_Uncertainty {uncertainty:.3f}\t'
                    f'Mem {memory_used:.0f}MB')

        avg_epoch_time += time.time() - start

        if scheduler is not None:
            scheduler.step()
    
    logger.info(f"Avg {args.epoch} Epoch training takes {datetime.timedelta(seconds=int(avg_epoch_time // args.epoch))}")


def eval_model(model, dataloader, args, logger):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()


    end = time.time()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(dataloader):
            input, target = input.to(args.device), target.to(args.device)

            output, _ = model(input)

            loss = criterion(output, target)

            # Measure uncertainty
            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = args.num_classes / torch.sum(alpha, dim=1, keepdim=True)
            uncertainty = torch.mean(uncertainty)

           # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Val: [{batch_idx:{int(math.log10(len(dataloader)))+1}}/{len(dataloader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Avg_Uncertainty {uncertainty:.3f}\t'
                    f'Mem {memory_used:.0f}MB')
        
        logger.info(f'[Val Avg]: Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    
    return acc1_meter.avg

def eval_single_image(model, img_path, args):
    img = Image.open(img_path)

    num_classes = 10
    plt.cla()

    t = []
    size = int((256 / 224) * args.input_size)
    t.append(transforms.Resize(size, interpolation=3)) # to maintain same ratio w.r.t. 224 images
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize((0.1307,),(0.3081,)))
    trans_mnist = transforms.Compose(t)

    img_tensor = trans_mnist(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    img_variable = img_variable.to(args.device)

    output, _ = model(img_variable)
    evidence = relu_evidence(output)
    alpha = evidence + 1
    uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    _, preds = torch.max(output, 1)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    output = output.flatten()
    prob = prob.flatten()
    preds = preds.flatten()
    print("Predict:", preds[0])
    print("Probs:", prob)
    print("Uncertainty:", uncertainty)

    labels = np.arange(10)
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})

    plt.title("Classified as: {}, Uncertainty: {}".format(preds[0], uncertainty.item()))

    axs[0].set_title("Image")
    axs[0].imshow(img)
    axs[0].axis("off")

    axs[1].bar(labels, prob.cpu().detach().numpy(), width=0.5)
    axs[1].set_xlim([0, 9])
    axs[1].set_ylim([0, 1])
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Classification Probability")

    plt.tight_layout()

    save_path = img_path.replace(args.sample_path, args.output_path)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    plt.savefig(save_path, dpi = 300)
    plt.close()