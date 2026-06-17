# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist
import pdb

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from torch.cuda.amp import autocast
from icecream import ic
def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int,scaler, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    training_states={
        'epoch':epoch,
    }
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        extra={'training_states':training_states}
        #pdb.set_trace()
        #img_data, img_id, text_data, word_selection, text_data_s, text_info, text_info_s, target = batch
        img_data, img_id, text_data, word_selection, text_info, target = batch
        #pdb.set_trace()
        #text_info convert
        #sentence
        
        text_info_data = {}
        max_length = 0
        for i in range(0, len(text_info)):
            if len(text_info[i]['input_ids'][0]) > max_length:
                max_length = len(text_info[i]['input_ids'][0])

        txt_input_ids = torch.zeros([len(text_info), max_length], dtype = torch.int64)
        #txt_token_type_ids = torch.zeros([len(text_info), max_length], dtype = torch.int64)
        txt_attention_mask_ids = torch.zeros([len(text_info), max_length], dtype = torch.int64)
        #pdb.set_trace()
        for i in range(0, len(text_info)):
            txt_input_ids[i][0:len(text_info[i]['input_ids'][0])] = text_info[i]['input_ids'][0]
            #xt_token_type_ids[i][0:len(text_info[i]['token_type_ids'][0])] = text_info[i]['token_type_ids'][0]
            txt_attention_mask_ids[i][0:len(text_info[i]['attention_mask'][0])] = text_info[i]['attention_mask'][0]

        text_info_data['input_ids'] = txt_input_ids
        #text_info_data['token_type_ids'] = txt_token_type_ids
        text_info_data['attention_mask'] = txt_attention_mask_ids

        #pdb.set_trace()
        #sentence_s
        # text_info_data_s = {}
        # max_length_s = 0
        # for i in range(0, len(text_info_s)):
        #     if len(text_info_s[i]['input_ids'][0]) > max_length_s:
        #         max_length_s = len(text_info_s[i]['input_ids'][0])

        # txt_input_ids_s = torch.zeros([len(text_info_s), max_length_s], dtype = torch.int64)
        # txt_attention_mask_ids_s = torch.zeros([len(text_info_s), max_length_s], dtype = torch.int64)

        # for i in range(0, len(text_info_s)):
        #     txt_input_ids_s[i][0:len(text_info_s[i]['input_ids'][0])] = text_info_s[i]['input_ids'][0]
        #     #xt_token_type_ids[i][0:len(text_info[i]['token_type_ids'][0])] = text_info[i]['token_type_ids'][0]
        #     txt_attention_mask_ids_s[i][0:len(text_info_s[i]['attention_mask'][0])] = text_info_s[i]['attention_mask'][0]

        # text_info_data_s['input_ids'] = txt_input_ids_s
        # #text_info_data['token_type_ids'] = txt_token_type_ids
        # text_info_data_s['attention_mask'] = txt_attention_mask_ids_s

        # copy to GPU
        for k in text_info_data:
            text_info_data[k] = text_info_data[k].to(device) if text_info_data[k] is not None else None
        # for k in text_info_data_s:
        #     text_info_data_s[k] = text_info_data_s[k].to(device) if text_info_data_s[k] is not None else None
        #pdb.set_trace()
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        word_selection = word_selection.to(device)
        #text_data_s = text_data_s.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        # model forward
        with autocast(enabled=args.amp):
            #output = model(img_data, text_data, extra)
            #output, region_features, diffused_sentences, valid_info, vl_similarity = model(img_data, text_data, word_selection, text_data_s, text_info_data, text_info_data_s, target, device, extra)
            output, vl_similarity, denoised_sentence, valid_sentence = model(img_data, text_data, word_selection, text_info_data, target, device, extra)
            #pdb.set_trace()
            if type(output)==dict:
                loss_dict = loss_utils.trans_vg_with_pruning_loss(output, target)
                loss_sentence = torch.zeros((), device=device)
                loss_vl = torch.zeros((), device=device)
            else:
                loss_dict = loss_utils.trans_vg_loss_dif(output, target)
                loss_sentence = loss_utils.sentence_similarity(denoised_sentence, valid_sentence)
                #loss_vl = loss_utils.sentence_region(diffused_sentences, region_features)
                loss_vl = vl_similarity.sum() / (vl_similarity.size(0) * vl_similarity.size(1))
            losses = sum(loss_dict[k] for k in loss_dict.keys()) + loss_sentence.sum() + loss_vl
        #pdb.set_trace()
        if args.amp:
            scaler.scale(losses).backward()
        else:
            losses.backward()
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        extra_loss_dict = utils.reduce_dict(
            {
                'loss_total': losses.detach(),
                'loss_sentence': loss_sentence.mean().detach(),
                'loss_vl': loss_vl.detach(),
            }
        )
        loss_dict_reduced_unscaled = {k: v
                                      for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
            
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled, **extra_loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        #pdb.set_trace()
        img_data, img_id, text_data, word_selection, text_info, target = batch
        batch_size = img_data.tensors.size(0)
        #pdb.set_trace()
        text_info_data = {}
        max_length = 0
        for i in range(0, len(text_info)):
            if len(text_info[i]['input_ids'][0]) > max_length:
                max_length = len(text_info[i]['input_ids'][0])

        txt_input_ids = torch.zeros([len(text_info), max_length], dtype = torch.int64)
        #txt_token_type_ids = torch.zeros([len(text_info), max_length], dtype = torch.int64)
        txt_attention_mask_ids = torch.zeros([len(text_info), max_length], dtype = torch.int64)
        #pdb.set_trace()
        for i in range(0, len(text_info)):
            txt_input_ids[i][0:len(text_info[i]['input_ids'][0])] = text_info[i]['input_ids'][0]
            #xt_token_type_ids[i][0:len(text_info[i]['token_type_ids'][0])] = text_info[i]['token_type_ids'][0]
            txt_attention_mask_ids[i][0:len(text_info[i]['attention_mask'][0])] = text_info[i]['attention_mask'][0]

        text_info_data['input_ids'] = txt_input_ids
        #text_info_data['token_type_ids'] = txt_token_type_ids
        text_info_data['attention_mask'] = txt_attention_mask_ids

        #pdb.set_trace()
        #sentence_s
        # text_info_data_s = {}
        # max_length_s = 0
        # for i in range(0, len(text_info_s)):
        #     if len(text_info_s[i]['input_ids'][0]) > max_length_s:
        #         max_length_s = len(text_info_s[i]['input_ids'][0])

        # txt_input_ids_s = torch.zeros([len(text_info_s), max_length_s], dtype = torch.int64)
        # txt_attention_mask_ids_s = torch.zeros([len(text_info_s), max_length_s], dtype = torch.int64)

        # for i in range(0, len(text_info_s)):
        #     txt_input_ids_s[i][0:len(text_info_s[i]['input_ids'][0])] = text_info_s[i]['input_ids'][0]
        #     #xt_token_type_ids[i][0:len(text_info[i]['token_type_ids'][0])] = text_info[i]['token_type_ids'][0]
        #     txt_attention_mask_ids_s[i][0:len(text_info_s[i]['attention_mask'][0])] = text_info_s[i]['attention_mask'][0]

        # text_info_data_s['input_ids'] = txt_input_ids_s
        # #text_info_data['token_type_ids'] = txt_token_type_ids
        # text_info_data_s['attention_mask'] = txt_attention_mask_ids_s

        # copy to GPU
        for k in text_info_data:
            text_info_data[k] = text_info_data[k].to(device) if text_info_data[k] is not None else None
        # for k in text_info_data_s:
        #     text_info_data_s[k] = text_info_data_s[k].to(device) if text_info_data_s[k] is not None else None

        #img_data = img_data.to(device)
        #text_data = text_data.to(device)
        #target = target.to(device)
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        #text_data_s = text_data_s.to(device)
        target = target.to(device)
        
        #output, region_features, diffused_sentences, valid_info, vl_similarity = model(img_data, text_data, text_data_s, text_info_data, text_info_data_s, device, {})
        output = model(img_data, text_data, word_selection, text_info_data, target, device, {})
        #pdb.set_trace()
        miou, accu, accu_07 = eval_utils.trans_vg_eval_dif(output, target.expand(output.size(0),target.size(1)))
        #pdb.set_trace()
        metric_logger.update_v2('miou', float(miou.item()), batch_size)
        metric_logger.update_v2('accu', float(accu), batch_size)
        metric_logger.update_v2('accu_07', float(accu_07), batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    from torchvision import transforms
    from utils.box_utils import xywh2xyxy
    from PIL import Image, ImageDraw
    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        output = model(img_data, text_data,extra={})

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    
    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    
    return accuracy


@torch.no_grad()
def evaluate_dif(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, img_id, text_data, word_selection, text_info, target = batch
        batch_size = img_data.tensors.size(0)

        text_info_data = {}
        max_length = 0
        for i in range(0, len(text_info)):
            if len(text_info[i]['input_ids'][0]) > max_length:
                max_length = len(text_info[i]['input_ids'][0])

        txt_input_ids = torch.zeros([len(text_info), max_length], dtype=torch.int64)
        txt_attention_mask_ids = torch.zeros([len(text_info), max_length], dtype=torch.int64)
        for i in range(0, len(text_info)):
            txt_input_ids[i][0:len(text_info[i]['input_ids'][0])] = text_info[i]['input_ids'][0]
            txt_attention_mask_ids[i][0:len(text_info[i]['attention_mask'][0])] = text_info[i]['attention_mask'][0]

        text_info_data['input_ids'] = txt_input_ids
        text_info_data['attention_mask'] = txt_attention_mask_ids

        for k in text_info_data:
            text_info_data[k] = text_info_data[k].to(device) if text_info_data[k] is not None else None

        img_data = img_data.to(device)
        text_data = text_data.to(device)
        word_selection = word_selection.to(device)
        target = target.to(device)

        output = model(img_data, text_data, word_selection, text_info_data, target, device, {})
        miou, accu, accu_07 = eval_utils.trans_vg_eval_dif(output, target.expand(output.size(0), target.size(1)))
        metric_logger.update_v2('miou', float(miou.item()), batch_size)
        metric_logger.update_v2('accu', float(accu), batch_size)
        metric_logger.update_v2('accu_07', float(accu_07), batch_size)

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats
        
