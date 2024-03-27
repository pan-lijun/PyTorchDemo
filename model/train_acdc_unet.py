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

from torchvision.transforms import functional as TF


def train_model_with_gpu(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs, device):
    # 将模型移动到GPU（如果可用）
    print(device)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # 将模型设为训练模式
        running_loss = 0.0
        # 训练集
        for batch_idx, (data, target, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 检查是否有label信息，如果没有则返回全0输出及损失
            if label is not None:
                loss = crop_and_forward(data, target, label, model, device, criterion)

                # 计算梯度并更新权重
                loss.backward()
                optimizer.step()
            else:
                # 当label为None时，直接生成全0输出并计算损失，但不进行反向传播和优化
                output = torch.zeros((data.shape[0], model.n_classes, data.shape[2], data.shape[3]), device=device)
                softmax_output = F.softmax(output, dim=1).float()
                one_hot_target = F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float().to(device)
                loss = criterion(softmax_output, one_hot_target, multiclass=True)

            # 记录运行损失
            item_loss = loss.item() * data.size(0)
            running_loss += item_loss
            print(batch_idx, item_loss)

            # 每个epoch结束时，计算平均损失并输出
        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f'Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}')

        # 验证集
        model.eval()  # 将模型设为评估模式
        validating_loss = 0
        with torch.no_grad():  # 在验证阶段关闭自动求导以提高效率
            for batch_idx, (data, target, label) in enumerate(validation_dataloader):
                # 检查是否有label信息，如果有则进行crop_and_forward操作
                if label is not None:
                    loss = crop_and_forward(data, target, label, model, device, criterion)
                else:
                    # 当label为None时，直接生成全0输出并计算损失
                    output = torch.zeros((data.shape[0], model.n_classes, data.shape[2], data.shape[3]), device=device)
                    softmax_output = F.softmax(output, dim=1).float()
                    one_hot_target = F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float().to(device)
                    loss = criterion(softmax_output, one_hot_target, multiclass=True)

                # 记录运行损失
                item_loss = loss.item() * data.size(0)
                validating_loss += item_loss

            # 每个epoch结束时，计算平均损失并输出
        epoch_loss = validating_loss / len(validation_dataloader.dataset)
        print(f'Epoch {epoch + 1}, Validating Loss: {epoch_loss:.4f}')

        model.train()  # 验证后重新将模型切换回训练模式

    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, '../checkpoints/checkpoint_acdc_10_2024_03_26.pth')

    return model


def crop_and_forward(data: torch.Tensor, target: torch.Tensor, label: tuple, model: torch.nn.Module,
                     device: torch.device, criterion) -> torch.Tensor:
    classes = model.n_classes
    resized_dim = (128, 128)

    # 将data和model移动到设备上
    data = data.to(device)
    model = model.to(device)

    # 解析label元组，假设有(x1, y1, x2, y2)四个元素
    x1, y1, x2, y2 = label

    # 裁剪
    cropped_data = data[:, :, y1:y2, x1:x2]

    cropped_data_resized = F.interpolate(cropped_data,
                                         size=resized_dim,
                                         mode='nearest')

    # 将resize后的数据送入模型
    output = model(cropped_data_resized)

    # 输出尺寸还原至裁剪前大小
    output_original_size = F.interpolate(output,
                                         size=(cropped_data.shape[2], cropped_data.shape[3]),
                                         mode='nearest')

    # 创建一个全零张量用于存放还原后的输出
    full_size_output = torch.zeros((data.shape[0], classes, data.shape[2], data.shape[3]), device=device)

    full_size_output[:, :, y1:y2, x1:x2] = output_original_size

    # 计算损失
    softmax_output = F.softmax(full_size_output, dim=1).float()
    one_hot_target = F.one_hot(target, classes).permute(0, 3, 1, 2).float().to(device)
    loss = criterion(softmax_output, one_hot_target, multiclass=True)

    return loss


if __name__ == '__main__':
    train_img_path = '..\\data\\acdc\\training\\imgs'
    train_mask_path = '..\\data\\acdc\\training\\masks'
    train_label_path = '..\\data\\acdc\\training\\labels'
    validation_img_path = '..\\data\\acdc\\validating\\imgs'
    validation_mask_path = '..\\data\\acdc\\validating\\masks'
    validation_label_path = '..\\data\\acdc\\validating\\labels'

    dir_checkpoint = '../checkpoints'
    lr = 1e-3
    num_epochs = 10

    dataloader_params = {
        'batch_size': 1,  # 根据实际情况调整批次大小
        'shuffle': False,  # 是否在每个epoch开始时打乱数据顺序
        # 'num_workers': 4,  # 并行加载数据的工作进程数（根据机器CPU核心数合理设置）
        # 'pin_memory': True,  # 如果可能，将数据复制到CUDA设备内存中以加速数据传输
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # class_weights = torch.tensor([0.1, 2, 4, 1])
    # class_weights.to(device)
    # 创建数据加载器
    train_dataset = ACDCDateSet(train_img_path, train_mask_path, train_label_path)
    validation_dataset = ACDCDateSet(validation_img_path, validation_mask_path, validation_label_path)
    train_dataloader = DataLoader(dataset=train_dataset, **dataloader_params)
    validation_dataloader = DataLoader(dataset=validation_dataset, **dataloader_params)

    model = UNet(1, 4)
    # criterion = dice_score_weighted.dice_loss
    criterion = dice_score.dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train_model_with_gpu(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs,
                                 device)
    model.to('cpu')

    prediction_img_path = '../data/acdc/testing/imgs/patient101_frame01_slice09.png'
    mask_path = '../data/acdc/testing/masks/patient101_frame01_slice09.png'
    label_path = '../data/acdc/testing/labels/patient101_frame01_slice09.txt'
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
