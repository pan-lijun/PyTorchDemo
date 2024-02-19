from model.unet import UNet
import torch
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image
from model.ACDCDataset import ACDCDateSet

model = UNet(1, 4)

model_path = '../checkpoints/checkpoint.pth'
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

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
