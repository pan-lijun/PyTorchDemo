import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class ACDCDateSet(Dataset):
    def __init__(self, img_path, mask_path):
        self.img_path = img_path
        self.mask_path = mask_path

        self.img_ids = []
        self.mask_ids = []

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
        # 获取img和mask的路径
        img_file_path = os.path.join(self.img_path, img_name)
        mask_file_path = os.path.join(self.mask_path, mask_name)

        # 读取img和mask
        img_pil = Image.open(img_file_path).convert('L')
        mask_pil = Image.open(mask_file_path).convert('L')

        # 图像预处理
        img_tensor = self.preprocess(img_pil, False)
        mask_tensor = self.preprocess(mask_pil, True)

        return img_tensor, mask_tensor

    @staticmethod
    def preprocess(pil_img, is_mask):
        pil_img = pil_img.resize((256, 256), resample=Image.NEAREST if is_mask else Image.BICUBIC)
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
    # i = 0
    for batch_idx, (images, masks) in enumerate(data_loader):
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
