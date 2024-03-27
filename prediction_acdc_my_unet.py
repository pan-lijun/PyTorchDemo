from model.my_unet import MyUNet
import torch
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image
from model import dice_loss, dice_score, dice_score_weighted
from model.ACDCDataset import ACDCDateSet
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyUNet(1, 4)

model_path = '.\\checkpoints\\checkpoint_acdc_my_unet_100.pth'
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
model.set_train_mode(False)
model.merge_weights()
# model.to(device)

test_img_path = '.\\temp\\acdc\\prediction\\imgs'
test_mask_path = '.\\temp\\acdc\\prediction\\masks'
save_path = '.\\temp\\acdc\\prediction\\prediction_my_unet'

dataloader_params = {
    'batch_size': 1,  # 根据实际情况调整批次大小
    'shuffle': False,  # 是否在每个epoch开始时打乱数据顺序
    # 'num_workers': 4,  # 并行加载数据的工作进程数（根据机器CPU核心数合理设置）
    # 'pin_memory': True,  # 如果可能，将数据复制到CUDA设备内存中以加速数据传输
}

# 创建数据加载器
test_dataset = ACDCDateSet(test_img_path, test_mask_path)
test_dataloader = DataLoader(dataset=test_dataset, **dataloader_params)
dice_score_fn = dice_score_weighted.multiclass_dice_coeff
dice_score_list = []


def show_prediction_img(data, outputs):
    # global outputs, target
    outputs = outputs.squeeze(0)
    outputs = torch.softmax(outputs, dim=0)
    max_probs, argmax_indices = outputs.max(dim=0)
    argmax_indices = argmax_indices.to(torch.int)
    prediction_array = argmax_indices.detach().numpy()
    # print(data.shape)
    img_tensor = data.squeeze(0).squeeze(0)
    img_array = img_tensor.detach().numpy()
    # rgb_img = np.dstack([img_array] * 3)
    # 假设你的归一化灰度图像numpy数组 img_array 和 标签预测数组 prediction_array 形状均为 (256, 256)
    height, width = img_array.shape
    # 将灰度图转换为RGB图像，每个像素都是相同的灰度值
    rgb_img = np.dstack([img_array] * 3).astype(np.float32)
    # 创建一个与原图大小相同的RGBA图像列表，用于分别添加红色、绿色和蓝色标记
    color_overlays = [
        np.zeros((height, width, 4), dtype=np.float32),
        np.zeros((height, width, 4), dtype=np.float32),
        np.zeros((height, width, 4), dtype=np.float32)
    ]
    # 定义颜色及透明度
    colors = {
        1: [1, 0, 0, 0.5],  # 红色
        2: [0, 1, 0, 0.5],  # 绿色
        3: [0, 0, 1, 0.5]  # 蓝色
    }
    # 对于每个颜色标签，将其对应的颜色叠加到对应的覆盖层上
    for label, color in colors.items():
        non_transparent_indices = (prediction_array == label)
        color_overlays[label - 1][non_transparent_indices, :] = color
    # 将颜色叠加到原图上，考虑到透明度的影响
    rgb_img_with_colors = rgb_img.copy().astype(np.float32)
    for overlay in color_overlays:
        non_transparent_indices = overlay[..., 3] > 0
        original_color = rgb_img[non_transparent_indices]
        new_color = overlay[non_transparent_indices, :3]
        alpha = overlay[non_transparent_indices, 3][:, np.newaxis]
        blended_color = (1 - alpha) * original_color + alpha * new_color
        rgb_img_with_colors[non_transparent_indices] = blended_color
    # 确保结果在[0, 255]范围内并转换为uint8类型以供显示
    rgb_img_with_colors = np.clip(rgb_img_with_colors, 0, 1) * 255
    rgb_img_with_colors = rgb_img_with_colors.astype(np.uint8)
    # 显示图像
    plt.imshow(rgb_img_with_colors)
    plt.show()

    # plt.savefig(f'{save_path}\\{batch_idx}.png')
    rgb_pil_image = Image.fromarray(rgb_img_with_colors.astype('uint8'), 'RGB')

    # 保存图像
    rgb_pil_image.save(f'{save_path}\\{batch_idx}.png')
    # print(f'{save_path}\\{batch_idx}.png')


for batch_idx, (data, target) in enumerate(test_dataloader):
    # 将数据和目标也移动到GPU
    # data, target = data.to(device), target.to(device)
    outputs = model(data)
    dice_score = dice_score_fn(F.softmax(outputs, dim=1).float(),
                               F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float())
    print(f"index : {batch_idx} ======> dice_score: {dice_score}")
    dice_score_list.append(dice_score.item())

    show_prediction_img(data, outputs)
    #
    # if batch_idx == 20:
    #     break
# 将dice_score_list转换为numpy数组
dice_scores_array = np.array(dice_score_list)

print(dice_score_list)

# 计算平均值（均值）
mean_value = dice_scores_array.mean()

# 计算标准差
std_deviation = dice_scores_array.std()

print(f"mean: {mean_value}, std_deviation: {std_deviation}")
