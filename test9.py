import os
import matplotlib.pyplot as plt
from skimage.io import imread

# 定义图像文件夹和标签文件夹路径
image_folder = 'ACDC_nii/images'
label_folder = 'ACDC_nii/labels'

# 获取两个文件夹下的所有文件名，并按顺序排列
image_files = sorted(os.listdir(image_folder))
label_files = sorted(os.listdir(label_folder))


def display_images_and_labels(index):
    # 加载当前索引对应的图像和标签
    img = imread(os.path.join(image_folder, image_files[index]))
    lbl = imread(os.path.join(label_folder, label_files[index]))

    # 显示图像和标签
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(lbl, cmap='gray')  # 根据实际情况选择合适的颜色映射
    axs[1].set_title('Label Image')
    axs[1].axis('off')

    plt.show()


# 遍历并询问用户是否要查看下一张
for i in range(len(image_files)):
    print(f"Showing imgs pair {i + 1}/{len(image_files)}")
    display_images_and_labels(i)

    # 用户输入y继续，其他字符退出
    user_input = input("Press '1' to view the next pair, any other key to exit: ")
    if user_input.lower() != '1':
        break
