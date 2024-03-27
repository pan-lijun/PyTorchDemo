import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.ACDCDataset import ACDCDateSet
import model
from model.unet import UNet


def one_hot_encoding(target, num_classes):
    # 将 (batch_size, height, width) 形状的目标标签转换为 one-hot 编码
    return torch.eye(num_classes)[target.reshape(-1)].reshape(target.shape[0], -1, *target.shape[1:])


def dice_loss(prediction, target, num_classes, smooth=1e-5):
    # 首先进行 one-hot 编码
    target_one_hot = one_hot_encoding(target, num_classes)

    # print("target_one_hot:", target_one_hot.shape)

    # 计算 intersection 和 union
    intersection = torch.sum(prediction * target_one_hot, dim=(2, 3))
    union = torch.sum(prediction + target_one_hot, dim=(2, 3)) - intersection

    # 计算 Dice coefficient per class
    dice_coefficient = (2. * intersection + smooth) / (union + smooth)

    # 返回平均Dice损失
    return 1. - dice_coefficient.mean()


# 假设已有的预测结果和真实标签（非 one-hot 编码）
# preds = torch.randn(4, 5, 3, 3)  # 4个样本，5个类别，图像尺寸为256x256
# true_labels = torch.randint(0, 5, size=(4, 3, 3))  # 真实标签
#
# print("preds:", preds)
# print("true_labels:", true_labels)
#
# num_classes = 5  # 总类别数
# loss = dice_loss(preds, true_labels, num_classes=num_classes)
#
# print("loss:", loss)

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
    predictions = nn.functional.softmax(predictions, dim=1)

    print(predictions.shape)
    print(predictions)

    loss = dice_loss(predictions, masks, num_classes=4)

    print(loss)
