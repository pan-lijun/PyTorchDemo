import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.ACDCDataset import ACDCDateSet
import model
from model.unet import UNet
import torch.nn.functional as F


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # print('dice_coeff')
    # print('input', input.shape)
    # print('target', target.shape)
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    # print(dice.shape)
    return dice.mean()


def apply_class_weights(input: Tensor, target: Tensor, class_weights: Tensor):
    assert input.size() == target.size()
    assert input.dim() == 4 and class_weights.dim() == 1

    # 将类别权重扩展到与target相同的形状（B,C,H,W）
    expanded_class_weights = class_weights.view(1, -1, 1, 1).expand_as(target)

    # 对于每个像素位置，根据其所在类别索引应用相应的权重
    weighted_target = target * expanded_class_weights.gather(dim=1, index=target.argmax(dim=1, keepdim=True))

    return weighted_target


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, class_weights: Tensor, multiclass: bool = False):
    weighted_masks = apply_class_weights(torch.zeros_like(target), target, class_weights)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, weighted_masks, reduce_batch_first=True)


if __name__ == '__main__':
    img_path = '../data/acdc/imgs'
    mask_path = '../data/acdc/masks'

    dataloader_params = {
        'batch_size': 4,  # 根据实际情况调整批次大小
        'shuffle': True,  # 是否在每个epoch开始时打乱数据顺序
        # 'num_workers': 4,  # 并行加载数据的工作进程数（根据机器CPU核心数合理设置）
        'pin_memory': True,  # 如果可能，将数据复制到CUDA设备内存中以加速数据传输
    }
    dataset = ACDCDateSet(img_path, mask_path)
    data_loader = DataLoader(dataset=dataset, **dataloader_params)

    for images, masks in data_loader:
        # 这里 images 和 masks 分别是第一个样本的图像和掩模
        break

    model = UNet(1, 4)

    predictions = model(images)
    # predictions = nn.functional.softmax(predictions, dim=1)
    predictions = F.softmax(predictions, dim=1).float()
    masks = F.one_hot(masks, model.n_classes).permute(0, 3, 1, 2).float()

    print(predictions.shape)
    # print(predictions)
    print(masks.shape)
    class_weights = torch.tensor([0.1, 1, 1.5, 1])
    loss = dice_loss(predictions, masks, class_weights, multiclass=True)

    print(loss)
    # predictions = torch.Tensor([[[[0, 1], [2, 3]]]]).float()
    # masks = torch.Tensor([[[[1, 0], [2, 3]]]]).float()
    # loss = dice_loss(predictions, masks, multiclass=True)
    # print(loss)
