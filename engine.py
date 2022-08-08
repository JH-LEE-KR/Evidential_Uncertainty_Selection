import time
import math
import os
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from timm.utils import accuracy, AverageMeter

from utils import *

import matplotlib.pyplot as plt


def train_model(model, criterion, train_loader, val_loader, optimizer, scheduler, args, logger):
    model.train()
        
    max_acc = 0
    max_epoch = 0
    
    avg_epoch_time = 0
    train_loss = []
    train_acc1 = []

    for epoch in range(0, args.epoch):
        keep_rate = None
        if epoch < args.warmup_epoch:
            keep_rate = [1.0] * model.depth

        num_steps = len(train_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()

        start = time.time()
        end = time.time()

        for batch_idx, (input, target) in enumerate(train_loader):
            input, target = input.to(args.device), target.to(args.device)

            output, _ = model(input, keep_rate)

            loss = criterion(output, target)

            if args.uncertainty_loss:
                # Measure uncertainty
                evidence = F.relu(output)
                alpha = evidence + 1
                uncertainty = args.num_classes / torch.sum(alpha, dim=1, keepdim=True)
                uncertainty = torch.mean(uncertainty)

                loss = loss + args.uncertainty_weight * uncertainty
            
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
                    f'Train: Epoch[{epoch+1:{int(math.log10(args.epoch))+1}}/{args.epoch}][{batch_idx:{int(math.log10(len(train_loader)))+1}}/{num_steps}]\t'
                    f'Eta {datetime.timedelta(seconds=int(etas))}\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        avg_epoch_time += time.time() - start

        if scheduler is not None:
            scheduler.step()
        
        # test and save value for plotting
        acc, loss = eval_model(model, val_loader, args, logger)
        train_loss.append(loss)
        train_acc1.append(acc)

        if acc > max_acc:
            max_acc = acc
            max_epoch = epoch
            try:
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
            except OSError:
                print ('=> Error: Failed to create directory')
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
    
    logger.info(f"Avg {args.epoch} Epoch training takes {datetime.timedelta(seconds=int(avg_epoch_time // args.epoch))}")
    
    plt.plot(range(args.epoch), train_loss)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(0, args.epoch, 5))
    plt.savefig(os.path.join(args.output_path, "train_loss.png"))
    plt.clf()
    plt.plot(range(args.epoch), train_acc1)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(range(0, args.epoch, 5))
    plt.savefig(os.path.join(args.output_path, "train_acc1.png"))

    logger.info(f'[Best Model]: Epoch {max_epoch + 1}\tAcc@1 {max_acc:.3f}')

    return train_loss, train_acc1

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
            evidence = F.relu(output)
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
    
    return acc1_meter.avg, loss_meter.avg
