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

from DiffRate.utils import ste_min

import numpy as np


def rep_double(tensor):
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            if torch.isinf(tensor[i][j]):
                tensor[i][j] = tensor[i][j-1]
    return tensor


# lat_lut = torch.tensor(np.load("layer_lat.npy"), device = torch.device('cuda:0'))
# inf_tensor = torch.full((197,197), float("inf"), dtype=lat_lut.dtype)
# original_slice = (slice(0, lat_lut.shape[0]), slice(0, lat_lut.shape[1]))
# inf_tensor[original_slice] = lat_lut
# inf_tensor = rep_double(inf_tensor.clone())
# lat_lut = torch.tensor(inf_tensor, device = torch.device('cuda:0'), dtype=torch.half)

def train_one_epoch(loss_fn, writer, model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,logger=None,target_flops=3.0,warm_up=False, target_etrr=25.0, target_thru=75.0, target_batch_size=8):
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
            outputs, flops, sched, dec, etrr, merge_kept_num_prob = model(samples)
            
            # Compute loss
            loss, loss_cls, thru_loss = loss_fn(outputs, targets, etrr)

            # loss_cls = criterion(outputs, targets)
            # loss_flops = ((flops/1e9)-target_flops)**2
            
           
            # etrr_loss = ((etrr - target_etrr)) **2

            # loss_sms = sum(model._diffrate_info["merge_decision"]) ** 2 #torch.log((sum(dec)+1) **2) # allows to play with few merging locations withuot increasing loss too much
            # # print(loss_tome)
            # # loss_etrr_per_merge = merge_locs / etrr_val
            # alpha = 50 # 50 #0.1 #10
            # beta = 100 #10 #50 #10
            # gamma = 10 #100 #1 #10 #0.1

            # # loss = loss_lat
            # loss = thru_loss + loss_cls #10 * loss_cls + etrr_loss #gamma * loss_cls + beta * etrr_loss + alpha * loss_tome
            # # if data_iter_step % 2 == 0:
            # #     loss = 10 * thru_loss + loss_cls + 0.001 * loss_sms
            # # else:
            # #     loss = 10 * thru_loss + loss_cls
                

            # # loss = lamb * loss_flops #beta * etrr_loss
    
            # # loss =  beta * etrr_loss + alpha * loss_tome # + gamma * loss_cls 

            loss_cls_value = loss_cls.item()
            # loss_flops_value = loss_flops.item()

            # loss_sms_value = loss_sms
            
            
        writer.add_scalar("loss", loss, (epoch+1) * data_iter_step)
        writer.add_scalar("thru_loss", thru_loss, (epoch+1) * data_iter_step)
        # writer.add_scalar("etrr_loss", etrr_loss, (epoch+1) * data_iter_step)
        writer.add_scalar("loss_cls", loss_cls, (epoch+1) * data_iter_step)
        # writer.add_scalar("loss_sms", loss_sms, (epoch+1) * data_iter_step)
        # writer.add_scalar("loss_flops", loss_flops_value, (epoch+1) * data_iter_step)

        merge_dec = model.get_dec()
        merge_prob = model.get_merge_prob()

        i = 1
        for dec, prob in zip(merge_dec, merge_prob):
            writer.add_scalar(f"merge_dec_{i}", dec, (epoch+1) * data_iter_step)
            writer.add_scalar(f"merge_prob_{i}", prob, (epoch+1) * data_iter_step)
            i+=1
        


        
        if not math.isfinite(loss_cls_value):
            logger.info("Loss is {}, stopping training".format(loss_cls_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.arch_parameters(), create_graph=is_second_order)
        
        writer.add_scalar("grad_norm", grad_norm, (epoch+1) * data_iter_step)
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
        metric_logger.update(loss=loss)
        # metric_logger.update(loss_etrr_per_merge=loss_etrr_per_merge)
        # metric_logger.update(loss_sms=loss_sms_value)
        # metric_logger.update(loss_lat=loss_lat)
        # metric_logger.update(thru=thru)
        metric_logger.update(thru_loss=thru_loss)
        metric_logger.update(etrr=etrr)
        # metric_logger.update(etrr_loss=etrr_loss)
        # metric_logger.update(loss_flops=loss_flops_value)
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
            output, flops, sched, dec, etrr, merge_kept_num_prob = model(images)
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