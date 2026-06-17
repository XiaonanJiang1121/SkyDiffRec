import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy
import pdb

def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, accu

def trans_vg_eval_dif(pred_boxes, gt_boxes):
    '''
    for i in range(0, len(batch_pred)):
        pdb.set_trace()
        batch_size = batch_pred[i].shape[0]
        pred_boxes = batch_pred[i]
        pred_boxes = xywh2xyxy(pred_boxes)
        pred_boxes = torch.clamp(pred_boxes, 0, 1)
        gt_boxes = xywh2xyxy(gt_boxes)
        iou = bbox_iou(pred_boxes, gt_boxes)
        accu = torch.sum(iou >= 0.5) / float(batch_size)
    '''
    #pdb.set_trace()
    batch_size = pred_boxes.shape[0]
    #pred_boxes = batch_pred[i]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    
    ious = torch.zeros((batch_size, pred_boxes.size(1)), requires_grad=False).cuda()
    for i in range(0, pred_boxes.size(1)):
        iou = bbox_iou(pred_boxes[:,i,:], gt_boxes)
        ious[:,i] = iou 
    #pdb.set_trace()
    max_ious = torch.max(ious, 1).values
    accu_05 = torch.mean((max_ious >= 0.5).float()).item()
    accu_07 = torch.mean((max_ious >= 0.7).float()).item()
    miou = torch.mean(max_ious)

    return miou, accu_05, accu_07

def trans_vg_eval_test(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)

    return accu_num
