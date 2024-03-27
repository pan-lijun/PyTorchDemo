import os
import cv2
import numpy as np

# # 标签文件夹路径
# label_folder_path = ".\\data\\msd\\training\\masks"
#
# # 遍历文件夹中的所有png文件
# for filename in os.listdir(label_folder_path):
#     if filename.endswith(".png"):  # 只处理PNG文件
#         label_image_path = os.path.join(label_folder_path, filename)
#
#         # 读取PNG标签图像
#         label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)
#
#         if label_image is not None:
#             height, width = label_image.shape[:2]
#
#             for row in range(height):
#                 for col in range(width):
#                     pixel_value = label_image[row, col]
#
#                     # 检查像素值是否为0或1
#                     if pixel_value != 0 and pixel_value != 1:
#                         print(f"在图像 {filename} 的位置 ({row}, {col}) 发现非0/1的像素值: {pixel_value}")
#         else:
#             print(f"无法读取PNG标签图像 {filename}，请检查文件是否存在并格式正确。")

# 读取标签图像路径
# label_path = ".\\data\\msd\\training\\masks\\P01-0080-labels.png"
#
# # 读取PNG标签图像并转换为灰度模式
# label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#
# if label_image is not None:
#     height, width = label_image.shape
#
#     # 将所有非零像素值设为1
#     label_image[label_image != 0] = 1
#
#     # 现在label_image中的像素值只有0和1
# else:
#     print("无法读取PNG标签图像，请检查文件是否存在并格式正确。")
#
# # 若需要保存处理后的图像
# cv2.imwrite("processed_label.png", label_image)
# print("finished")

# 指定原始标签文件夹路径
original_label_folder_path = ".\\data\\msd\\testing\\masks"

# 指定目标保存标签文件夹路径
processed_label_folder_path = ".\\data\\msd\\testing\\masks2"

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(processed_label_folder_path):
    os.makedirs(processed_label_folder_path)

# 遍历原始标签文件夹中的所有PNG文件
for filename in os.listdir(original_label_folder_path):
    if filename.endswith(".png"):  # 只处理PNG文件
        original_label_image_path = os.path.join(original_label_folder_path, filename)
        processed_label_image_path = os.path.join(processed_label_folder_path, filename)

        # 加载图像并转为灰度模式
        label_image = cv2.imread(original_label_image_path, cv2.IMREAD_GRAYSCALE)

        if label_image is not None:
            # 将所有非零像素（即255）设置为1
            label_image[label_image != 0] = 1

            # 保存处理后的图像到目标文件夹，使用相同的文件名
            cv2.imwrite(processed_label_image_path, label_image)
            print(f"已处理标签图像 {filename}，保存到 {processed_label_image_path}")
        else:
            print(f"无法读取PNG标签图像 {filename}，请检查文件是否存在并格式正确。")

print("已经全部完成")
