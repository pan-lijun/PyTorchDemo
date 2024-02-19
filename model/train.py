from pathlib import Path

import torch
from model import dice_loss, dice_score, dice_score_weighted
from model.ACDCDataset import ACDCDateSet
from torch.utils.data import Dataset, DataLoader
from model.unet import UNet
from PIL import Image
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import torch.nn.functional as F


def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()  # 将模型设为训练模式

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            # 前向传播
            optimizer.zero_grad()  # 清除梯度缓冲区
            outputs = model(data)

            # 计算损失
            loss = criterion(
                F.softmax(outputs, dim=1).float(),
                F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float(),
                class_weights,
                multiclass=True
            )

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 记录运行损失
            item_loss = loss.item() * data.size(0)
            running_loss += item_loss
            print(batch_idx, item_loss)

            # if batch_idx >= 2:
            #     break

        # 每个epoch结束时，计算平均损失并输出
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, '../checkpoints/checkpoint.pth')

    return model


def train_model_with_gpu(model, dataloader, criterion, optimizer, num_epochs, device):
    # 将模型移动到GPU（如果可用）
    print(device)
    model.to(device)

    model.train()  # 将模型设为训练模式

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            # 将数据和目标也移动到GPU
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()  # 清除梯度缓冲区
            outputs = model(data)

            # 计算损失
            loss = criterion(
                F.softmax(outputs, dim=1).float(),
                F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float().to(device),
                class_weights.to(device),  # 确保class_weights也在GPU上（如果适用）
                multiclass=True
            )

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 记录运行损失
            item_loss = loss.item() * data.size(0)
            running_loss += item_loss
            # print(batch_idx, item_loss)

            # if batch_idx >= 2:
            #     break

            # 每个epoch结束时，计算平均损失并输出
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, '../checkpoints/checkpoint_acdc.pth')

    return model


if __name__ == '__main__':
    img_path = '../data/acdc/training/imgs'
    mask_path = '../data/acdc/training/masks'
    dir_checkpoint = '../checkpoints'
    lr = 1e-3
    num_epochs = 4

    dataloader_params = {
        'batch_size': 8,  # 根据实际情况调整批次大小
        'shuffle': True,  # 是否在每个epoch开始时打乱数据顺序
        # 'num_workers': 4,  # 并行加载数据的工作进程数（根据机器CPU核心数合理设置）
        # 'pin_memory': True,  # 如果可能，将数据复制到CUDA设备内存中以加速数据传输
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor([0.1, 1.5, 3, 1])
    class_weights.to(device)

    dataset = ACDCDateSet(img_path, mask_path)
    data_loader = DataLoader(dataset=dataset, **dataloader_params)
    model = UNet(1, 4)
    criterion = dice_score_weighted.dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train_model_with_gpu(model, data_loader, criterion, optimizer, num_epochs, device)
    model.to('cpu')

    prediction_img_path = '../data/acdc/validating/imgs/patient100_frame13_slice04.png'
    mask_path = '../data/acdc/validating/masks/patient100_frame13_gt_slice04.png'
    img_pil = Image.open(prediction_img_path).convert('L')
    img_tensor = ACDCDateSet.preprocess(img_pil, False)
    img_batch = np.expand_dims(img_tensor, axis=0)
    img_tensor = torch.from_numpy(img_batch)
    # img_tensor.to(device)
    prediction = model(img_tensor)
    print(img_tensor.shape)
    print(prediction.shape)
    argmax = prediction.argmax(dim=1)
    print(argmax.shape)
    # print(argmax)
    gray_tensor = argmax.squeeze(0)
    print(gray_tensor.shape)
    # 将张量转换为numpy数组
    class_indices = gray_tensor.detach().cpu().numpy()

    # 创建一个颜色映射表，假设我们有4个类别
    cmap = ListedColormap(['black', 'red', 'green', 'blue'])

    # 显示带有颜色编码的图像
    plt.imshow(class_indices, cmap=cmap)
    plt.axis('off')  # 可选：关闭坐标轴
    plt.show()
