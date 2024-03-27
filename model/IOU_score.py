import torch
import torch.nn.functional as F
import numpy as np


def iou_score(output: torch.Tensor, target: torch.Tensor, num_classes: int):
    softmax_output = F.softmax(output, dim=1).float()
    predictions = torch.argmax(softmax_output, dim=1)

    # 确保predictions和target都是long类型（类别索引）
    predictions = predictions.long()
    target = target.long()

    # 初始化存储IoU的列表
    intersection = (predictions == target).float().sum()

    union = predictions.numel()

    iou = intersection / union + 1e-8

    return iou
