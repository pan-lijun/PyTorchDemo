import cv2
import numpy as np

# 加载原始灰度图像
original_image = cv2.imread('data/acdc/validating/imgs/patient100_frame13_slice04.png', cv2.IMREAD_GRAYSCALE)

# 将原始灰度图像转换为彩色图像，通过复制灰度值到三个通道
rgb_original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

# 加载灰度标签图像
label_image = cv2.imread('data/acdc/validating/masks/patient100_frame13_gt_slice04.png', cv2.IMREAD_GRAYSCALE)

# 定义颜色映射表（包括0）
color_mapping = {
    1: (0, 0, 255),  # 左心室
    2: (0, 255, 0),  # 心肌
    3: (255, 0, 0),  # 右心室
}

# 创建一个新的彩色图像，与原始图像尺寸相同
output_image = rgb_original_image.copy()

# 对于灰度值为0的区域，直接保留原灰度图像的颜色
mask_0 = (label_image == 0)
output_image[mask_0] = rgb_original_image[mask_0]

# 对于其他灰度值，根据颜色映射表进行替换
for label_value, color in color_mapping.items():
    if label_value != 0:
        mask = (label_image == label_value)
        output_image[mask] = color

cv2.imshow('Modified Image', output_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
