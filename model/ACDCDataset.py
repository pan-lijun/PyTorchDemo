import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


class ACDCDateSet(Dataset):
    def __init__(self, img_path, mask_path, label_path):
        self.img_path = img_path
        self.mask_path = mask_path
        self.label_path = label_path

        self.img_ids = []
        self.mask_ids = []
        # self.label_ids = []

        for filename in os.listdir(img_path):
            # 检查是否为文件而不是子目录
            if os.path.isfile(os.path.join(self.img_path, filename)):
                # 添加文件名到self.img_ids列表
                self.img_ids.append(filename)

        for filename in os.listdir(mask_path):
            # 检查是否为文件而不是子目录
            if os.path.isfile(os.path.join(self.mask_path, filename)):
                # 添加文件名到self.img_ids列表
                self.mask_ids.append(filename)

        assert len(self.img_ids) == len(self.mask_ids), "Image and Mask have different size"

        # for i in range(len(self.img_ids)):
        #     print(f'idx:{i},\n img_id: {self.img_ids[i]},\n mask_id: {self.mask_ids[i]}')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # 获取img和mask的名称
        img_name = self.img_ids[idx]
        mask_name = self.mask_ids[idx]

        filename = img_name.split('.')[0]
        label_file_path = os.path.join(self.label_path, filename + '.txt')
        label_file_path = label_file_path if os.path.isfile(label_file_path) else None

        # 获取img和mask的路径
        img_file_path = os.path.join(self.img_path, img_name)
        mask_file_path = os.path.join(self.mask_path, mask_name)

        # 读取img和mask
        img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
        # 获取图像的宽度和高度
        height, width = img.shape[:2]
        # print(img.shape)

        # 图像预处理
        img_tensor = self.preprocess(img, False)
        # print(img_tensor.shape)
        mask_tensor = self.preprocess(mask, True)
        # label_tensor = self.get_label_tensor(256, 256, label_file_path)
        class_id, label_tuple = self.get_label_tuple(height, width, label_file_path)

        return img_tensor, mask_tensor, label_tuple

    @staticmethod
    def preprocess(pil_img, is_mask):

        # pil_img = pil_img.resize((256, 256), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # pil_img = cv2.resize(pil_img, (256, 256), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC)

        img_numpy = np.array(pil_img)
        if is_mask:
            # 如果是mask，先跳过
            img_tensor = torch.from_numpy(img_numpy).long()
        else:
            # 如果是img，将图片归一化
            img_tensor = torch.from_numpy(img_numpy).float() / 255.0
            # 添加通道维度
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    @staticmethod
    def get_label_tensor(height, width, label_file_path):
        label_tensor = torch.zeros((height, width))
        if label_file_path is not None:
            with open(label_file_path, 'r') as f:
                line = f.readline().strip()
                class_id, x, y, w, h = [float(v) for v in line.strip().split(' ')]
                x_center, y_center = x * width, y * height
                bbox_width, bbox_height = w * width, h * height
                x_min = max(0, round(x_center - bbox_width / 2))
                y_min = max(0, round(y_center - bbox_height / 2))
                x_max = min(width - 1, round(x_center + bbox_width / 2))
                y_max = min(height - 1, round(y_center + bbox_height / 2))

                # 将指定区域内的像素值置为 1
                label_tensor[y_min:y_max + 1, x_min:x_max + 1] = 1

        return label_tensor

    @staticmethod
    def normalize_bbox_to_xyxy(x, y, w, h, height, width):
        # 将归一化中心点坐标和尺寸转换为归一化的左上角和右下角坐标
        center_x = x * width
        center_y = y * height
        half_width = w * width / 2
        half_height = h * height / 2

        # 左上角坐标
        x1 = max(0, center_x - half_width)
        y1 = max(0, center_y - half_height)

        # 右下角坐标
        x2 = min(width, center_x + half_width)
        y2 = min(height, center_y + half_height)

        # 返回归一化后的左上角和右下角坐标
        return math.ceil(x1), math.ceil(y1), math.floor(x2), math.floor(y2)

    # 在类方法中使用该函数
    def get_label_tuple(self, height, width, label_file_path):
        if label_file_path is not None:
            with open(label_file_path, 'r') as f:
                line = f.readline().strip()
                class_id, x, y, w, h = [float(v) for v in line.strip().split(' ')]

                # 转换坐标
                x1, y1, x2, y2 = self.normalize_bbox_to_xyxy(x, y, w, h, height, width)

                # 返回包含类别ID以及左上角和右下角坐标的元组
                bbox = (x1, y1, x2, y2)

                return class_id, bbox
        else:
            return None, (0, 0, 0, 0)


if __name__ == '__main__':
    img_path = '../data/acdc/training/imgs'
    mask_path = '../data/acdc/training/masks'
    label_path = '../data/acdc/training/labels'

    dataloader_params = {
        'batch_size': 1,  # 根据实际情况调整批次大小
        'shuffle': True,  # 是否在每个epoch开始时打乱数据顺序
        # 'num_workers': 4,  # 并行加载数据的工作进程数（根据机器CPU核心数合理设置）
        'pin_memory': True,  # 如果可能，将数据复制到CUDA设备内存中以加速数据传输
    }
    dataset = ACDCDateSet(img_path, mask_path, label_path)
    data_loader = DataLoader(dataset=dataset, **dataloader_params)
    # i = 0
    for batch_idx, (images, masks, label) in enumerate(data_loader):
        print(f"Batch {batch_idx}:")
        print("Images shape:", images.shape)
        print("Masks shape:", masks.shape)

        # # 每次显示前两张图片
        # plt.subplot(1, 2, 1)
        # plt.imshow(images[0].numpy(), cmap='gray')
        # plt.title('Image')
        # plt.subplot(1, 2, 2)
        # plt.imshow(masks[0].numpy(), cmap='gray')
        # plt.title('Mask')
        # plt.show()
    #
    #     i += 1
    #     if i > 10:
    #         break
