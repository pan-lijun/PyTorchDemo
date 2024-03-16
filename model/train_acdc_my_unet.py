from pathlib import Path
import torch
from model import dice_loss, dice_score, dice_score_weighted
from model.ACDCDataset import ACDCDateSet
from torch.utils.data import Dataset, DataLoader
from model.unet import UNet
from model.my_unet import MyUNet
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


def train_model_with_gpu(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs, device):
    # 将模型移动到GPU（如果可用）
    print(device)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # 将模型设为训练模式
        running_loss = 0.0
        # 训练集
        for batch_idx, (data, target) in enumerate(train_dataloader):
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
        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f'Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}')

        # 验证集
        model.eval()  # 将模型设为评估模式
        validating_loss = 0
        for batch_idx, (data, target) in enumerate(validation_dataloader):
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

            # # 反向传播和优化
            # loss.backward()
            # optimizer.step()

            # 记录运行损失
            item_loss = loss.item() * data.size(0)
            validating_loss += item_loss

            # 每个epoch结束时，计算平均损失并输出
        epoch_loss = validating_loss / len(validation_dataloader.dataset)
        print(f'Epoch {epoch + 1}, Validating Loss: {epoch_loss:.4f}')

        model.train()  # 验证后重新将模型切换回训练模式

    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, '../checkpoints/checkpoint_acdc_my_unet_40.pth')

    return model


if __name__ == '__main__':
    train_img_path = '..\\data\\acdc\\training\\imgs'
    train_mask_path = '..\\data\\acdc\\training\\masks'
    validation_img_path = '..\\data\\acdc\\validating\\imgs'
    validation_mask_path = '..\\data\\acdc\\validating\\masks'
    dir_checkpoint = '../checkpoints'
    lr = 1e-3
    num_epochs = 40

    dataloader_params = {
        'batch_size': 4,  # 根据实际情况调整批次大小
        'shuffle': True,  # 是否在每个epoch开始时打乱数据顺序
        # 'num_workers': 4,  # 并行加载数据的工作进程数（根据机器CPU核心数合理设置）
        # 'pin_memory': True,  # 如果可能，将数据复制到CUDA设备内存中以加速数据传输
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor([0.1, 2, 4, 1])
    class_weights.to(device)
    # 创建数据加载器
    train_dataset = ACDCDateSet(train_img_path, train_mask_path)
    validation_dataset = ACDCDateSet(validation_img_path, validation_mask_path)
    train_dataloader = DataLoader(dataset=train_dataset, **dataloader_params)
    validation_dataloader = DataLoader(dataset=validation_dataset, **dataloader_params)

    model = MyUNet(1, 4)
    # criterion = dice_score_weighted.dice_loss
    criterion = dice_score_weighted.dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train_model_with_gpu(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs,
                                 device)
    model.to('cpu')

    prediction_img_path = '../data/acdc/testing/imgs/patient101_frame01_slice09.png'
    mask_path = '../data/acdc/testing/masks/patient101_frame01_gt_slice09.png'
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
