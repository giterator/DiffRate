# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils



def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,logger=None,target_flops=3.0,warm_up=False, target_etrr=25.0):
    model.train(set_training_mode)
    # model.eval()
    # for name, param in model.named_parameters():
    #     print(name)
    #     if (not "selected_probability" in name) or (not "merge_prob" in name):  # Check if the parameter is a weight
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
            
    # model.train(False)      # finetune
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr_weight', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_architecture', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    logger.info_freq = 10
    compression_rate_print_freq = 100
    
    warm_up_epoch = 1     
    
    if warm_up and epoch < warm_up_epoch:   # for stable training and better performance
        lamb = 0
    else:
        lamb = 5

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, logger.info_freq, header,logger)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, flops, sched, dec, etrr = model(samples)
            loss_cls = criterion(outputs, targets)
            loss_flops = ((flops/1e9)-target_flops)**2
            
            # clean_sched = [197]
            # for k in sched:
            #     clean_sched.append(int(k.item()))
            
            # merge_locs = 0
            # reduction_sched = []
            # for i in range(len(clean_sched)-1):
            #     r = clean_sched[i] - clean_sched[i+1]
            #     reduction_sched.append(r)
            #     if r > 0:
            #         merge_locs += 1

            # clean_sched = []
            # for k in sched:
            #     clean_sched.append(int(k.item()))

            # reduction_sched = []
            # inp_tok = 197
            # merge_locs = 0
            # for k in clean_sched:
            #     r = inp_tok - k
            #     if r > 0:
            #         merge_locs += 1
            #         inp_tok -= r
            #         reduction_sched.append(r)
            #     else:
            #         reduction_sched.append(0)

            # loss_tome = (merge_locs / 12.0) **2

            # loss_tome = (sum(model.get_dec()) / 12.0) **2
            


            # def etrr(sched):
            #     sum = 0
            #     for i in range(len(sched)):
            #         layer = i+1
            #         sum += (sched[i] * (12-layer))
            #     etrr = 100 * sum/(197*12)
            #     return etrr
            
            # etrr_val = etrr(reduction_sched)
            etrr_loss = ((etrr - target_etrr)) **2

            # print(sched)
            # Loss terms MUST have grad func
            # print(type(sched))

            # loss = lamb * loss_flops + loss_cls
            # loss_tome = (sum(dec) / 12.0) **2
            loss_tome = torch.log((sum(dec)+1) **2) # allows to play with few merging locations withuot increasing loss too much
            # print(loss_tome)
            # loss_etrr_per_merge = merge_locs / etrr_val
            alpha = 1 # 50 #0.1 #10
            beta = 1 #10 #50 #10
            gamma = 0.1 #1 #10 #0.1

            if data_iter_step % 5 == 0:
                loss =  gamma * loss_cls + alpha * loss_tome
            else:
                loss =  gamma * loss_cls + beta * etrr_loss #lamb * loss_flops #

            # loss = lamb * loss_flops #beta * etrr_loss
    
            # loss =  beta * etrr_loss + alpha * loss_tome # + gamma * loss_cls 

            loss_cls_value = loss_cls.item()
            loss_flops_value = loss_flops.item()

            loss_tome_value = loss_tome
            
            
        
        
        if not math.isfinite(loss_cls_value):
            logger.info("Loss is {}, stopping training".format(loss_cls_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.arch_parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()

        if data_iter_step%compression_rate_print_freq == 0:
            if hasattr(model, 'module'):  # for DDP 
                # prune_kept_num, merge_kept_num = model.module.get_kept_num()
                merge_kept_num = model.module.get_kept_num()
                merge_dec = model.module.get_dec()
                merge_prob = model.module.get_merge_prob()
            else:
                # prune_kept_num, merge_kept_num = model.get_kept_num()
                merge_kept_num = model.get_kept_num()
                merge_dec = model.get_dec()
                merge_prob = model.get_merge_prob()
            # logger.info(f'prune kept number:{prune_kept_num}')
            logger.info(f'merge kept number:{merge_kept_num}')
            logger.info(f'merge decision:{merge_dec}')
            logger.info(f'merge prob:{merge_prob}')


        metric_logger.update(loss_cls=loss_cls_value)
        # metric_logger.update(loss_etrr_per_merge=loss_etrr_per_merge)
        metric_logger.update(loss_tome=loss_tome_value)
        metric_logger.update(etrr=etrr)
        metric_logger.update(etrr_loss=etrr_loss)
        metric_logger.update(loss_flops=loss_flops_value)
        metric_logger.update(flops=flops/1e9)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(lr_architecture=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device,logger=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    for images, target in metric_logger.log_every(data_loader, 10, header,logger):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, flops, sched, dec, etrr = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        torch.cuda.synchronize()

        batch_size = images.shape[0]
        metric_logger.update(flops=flops/1e9)
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if hasattr(model, 'module'):  # for DDP 
        # prune_kept_num, merge_kept_num = model.module.get_kept_num()
        merge_kept_num = model.module.get_kept_num()
        merge_dec = model.module.get_dec()
    else:
        # prune_kept_num, merge_kept_num = model.get_kept_num()
        merge_kept_num = model.get_kept_num()
        merge_dec = model.get_dec()
    # logger.info(f'prune kept number:{prune_kept_num}')
    logger.info(f'merge kept number:{merge_kept_num}')
    logger.info(f'merge decision:{merge_dec}')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} flops {flops.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, flops=metric_logger.flops))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}