import os
import cv2
import numpy as np

# 定义灰度值与新值的映射关系
value_mapping = {
    0: 0,
    85: 1,
    170: 2,
    255: 3,
}

input_path = './ACDC_nii/test_labels/masks'
output_path = './data/acdc/testing/masks'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 遍历输入路径下所有的图像文件
for filename in os.listdir(input_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # 根据你的实际图像格式选择
        img_path = os.path.join(input_path, filename)

        # 加载灰度标签图像
        label_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 将原始灰度值替换为新值
        new_label_image = np.zeros_like(label_image, dtype=np.uint8)
        for old_value, new_value in value_mapping.items():
            new_label_image[label_image == old_value] = new_value

        # 构建输出文件名和路径，保留原始文件名
        output_filename = os.path.basename(filename)  # 使用原始文件名
        output_img_path = os.path.join(output_path, output_filename)

        # 保存处理后的图像
        cv2.imwrite(output_img_path, new_label_image)
