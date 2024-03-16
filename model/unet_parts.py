import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.max_pool_conv(x)


class MyConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.train_mode = True  # 初始化训练模式为True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.merged_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        if self.train_mode:
            output1 = self.conv1(x)
            output2 = self.conv2(x)  # 注意输出形状会因为卷积核大小不同而变化
            output3 = self.conv3(x)
            true_output = output1 + output2 + output3
        else:
            true_output = self.merged_conv(x)

        return true_output

    def set_train_mode(self, is_train):
        self.train_mode = is_train

    def merge_weights(self):
        # 获取权重数据
        weights1 = self.conv1.weight.data
        weights2 = self.conv2.weight.data
        weights3 = self.conv3.weight.data
        true_weights = weights1.clone()  # 先复制一份原始权重
        true_weights[:, :, :, 1] += weights2.squeeze(dim=-1)
        true_weights[:, :, 1, :] += weights3.squeeze(dim=2)
        self.merged_conv.weight.data.copy_(true_weights)


class MyDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.train_mode = True
        self.double_conv = nn.Sequential(
            MyConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MyConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

    def set_train_mode(self, is_train):
        self.train_mode = is_train
        for module in self.double_conv.children():
            if isinstance(module, MyConv):
                module.set_train_mode(is_train)

    def merge_weights(self):
        # 遍历Sequential模块中的子模块，找到MyConv实例并调用其merge_weights方法
        for module in self.double_conv.children():
            if isinstance(module, MyConv):
                module.merge_weights()


class MyDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.train_mode = True
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MyDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.max_pool_conv(x)

    def set_train_mode(self, is_train):
        self.train_mode = is_train
        self.max_pool_conv[1].set_train_mode(is_train)  # 将训练模式设置传递给内部的 MyDoubleConv

    def merge_weights(self):
        self.max_pool_conv[1].merge_weights()  # 调用内部的 MyDoubleConv 的 merge_weights 方法来合并权重


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, y):
        """
        x为输入的特征图，y为下采样时的特征图
        :param x:
        :param y:
        :return:
        """
        x = self.up(x)

        diff_y = y.size()[2] - x.size()[2]
        diff_x = y.size()[3] - x.size()[3]

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x, y], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
