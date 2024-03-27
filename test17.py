import random
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance


def rotate_imgs(image_paths, label_paths, img_dir, mask_dir, rotation_degrees):
    for img_path, label_path in zip(image_paths, label_paths):
        # 加载原始图像和标签图像
        img = Image.open(img_path)
        label = Image.open(label_path)

        # 旋转图像
        rotated_img = img.rotate(rotation_degrees, expand=True)
        rotated_label = label.rotate(rotation_degrees, expand=True)

        # 获取新尺寸（可能因填充而变大）
        new_width, new_height = rotated_img.size

        # 裁剪到原来的尺寸以保持一致性
        cropped_rotated_img = rotated_img.crop((0, 0, img.width, img.height))
        cropped_rotated_label = rotated_label.crop((0, 0, label.width, label.height))

        img_name = img_path.split("\\")[-1]
        label_name = label_path.split("\\")[-1]

        # 保存旋转并裁剪后的图像和标签
        new_img_path = os.path.join(img_dir, f"rotated_{img_name}")
        new_label_path = os.path.join(mask_dir, f"rotated_{label_name}")

        # print(new_img_path, new_label_path)
        cropped_rotated_img.save(new_img_path)
        cropped_rotated_label.save(new_label_path)

        # img_array = np.array(img)
        # rotated_img_array = np.array(rotated_img)
        # label_array = np.array(labels)
        # rotated_label_array = np.array(rotated_label)
        #
        # # 创建一个新的figure，并设置子图布局为1行2列
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        #
        # # 显示第一张图片
        # axs[0].imshow(img_array)
        # axs[0].set_title('Image 1')
        #
        # # 显示第二张图片
        # axs[1].imshow(rotated_img_array)
        # axs[1].set_title('Image 2')
        #
        # # 移除所有子图的坐标轴（可选）
        # for ax in axs:
        #     ax.set_axis_off()
        #
        # # 显示图形
        # plt.show()
        #
        # # 创建一个新的figure，并设置子图布局为1行2列
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        #
        # # 显示第一张图片
        # axs[0].imshow(label_array, cmap='gray', vmin=0, vmax=3)
        # axs[0].set_title('Image 1')
        #
        # # 显示第二张图片
        # plt.imshow(rotated_label_array, cmap='gray', vmin=0, vmax=3)
        # axs[1].set_title('Image 2')
        #
        # # 移除所有子图的坐标轴（可选）
        # for ax in axs:
        #     ax.set_axis_off()
        #
        # # 显示图形
        # plt.show()

    print("Finished")


def adjust_brightness(image_paths):
    for img_path in image_paths:
        # 打开图像
        img = Image.open(img_path)

        # 创建亮度增强器对象
        enhancer = ImageEnhance.Brightness(img)

        # brightness_factor = 1 + random.uniform(-0.5, 0.5)
        brightness_factor = 0.5

        # 调整亮度
        brightened_img = enhancer.enhance(brightness_factor)

        img_name = img_path.split("\\")[-1]

        output_path = f"brightened_{img_name}"

        # 保存调整亮度后的图像
        brightened_img.save(output_path)

        img_array = np.array(img)
        brightened_img_array = np.array(brightened_img)

        # 创建一个新的figure，并设置子图布局为1行2列
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # 显示第一张图片
        axs[0].imshow(img_array)
        axs[0].set_title('Image 1')

        # 显示第二张图片
        axs[1].imshow(brightened_img_array)
        axs[1].set_title('Image 2')

        # 移除所有子图的坐标轴（可选）
        for ax in axs:
            ax.set_axis_off()

        # 显示图形
        plt.show()


def enhance_contrast(image_paths):
    for img_path in image_paths:
        # 打开图像
        img = Image.open(img_path)  # 将图像转换为灰度图像

        # 创建对比度增强器对象
        enhancer = ImageEnhance.Contrast(img)

        # 设置对比度增强因子（例如，1.5表示将对比度提高50%）
        contrast_factor = 1.5
        enhanced_img = enhancer.enhance(contrast_factor)

        img_name = img_path.split("\\")[-1]

        output_path = f"contrasted_{img_name}"

        # 保存增强后的图像
        enhanced_img.save(output_path)

        img_array = np.array(img)
        contrasted_img_array = np.array(enhanced_img)

        # 创建一个新的figure，并设置子图布局为1行2列
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # 显示第一张图片
        axs[0].imshow(img_array)
        axs[0].set_title('Image 1')

        # 显示第二张图片
        axs[1].imshow(contrasted_img_array)
        axs[1].set_title('Image 2')

        # 移除所有子图的坐标轴（可选）
        for ax in axs:
            ax.set_axis_off()

        # 显示图形
        plt.show()


def add_gaussian_noise(image_paths, mean=0, std_deviation=1):
    for img_path in image_paths:
        # 打开图像并转换为numpy数组
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)

        # 生成符合高斯分布的噪声
        # height, width = img_array.shape
        print(img_array.shape)
        noise = np.random.normal(mean, std_deviation, img_array.shape)

        # 将噪声加到图像上
        noisy_img_array = img_array + noise.astype(np.uint8)

        # 确保结果像素值在[0, 255]范围内
        noisy_img_array = np.clip(noisy_img_array, 0, 255)

        # 将处理后的numpy数组转回PIL图像格式
        noisy_img = Image.fromarray(noisy_img_array.astype('uint8'), 'L')

        img_name = img_path.split("\\")[-1]
        output_path = f"gaussian_noised_{img_name}"

        # 保存输出文件
        noisy_img.save(output_path)

        contrasted_img_array = np.array(noisy_img)

        # 创建一个新的figure，并设置子图布局为1行2列
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # 显示第一张图片
        axs[0].imshow(img_array)
        axs[0].set_title('Image 1')

        # 显示第二张图片
        axs[1].imshow(contrasted_img_array)
        axs[1].set_title('Image 2')

        # 移除所有子图的坐标轴（可选）
        for ax in axs:
            ax.set_axis_off()

        # 显示图形
        plt.show()


if __name__ == '__main__':
    # 假设你有图像路径和标签路径列表
    # image_paths = [".\\data\\data\\imgs\\patient001_frame01_slice02.png"]
    # label_paths = [".\\data\\data\\masks\\patient001_frame01_gt_slice02.png"]
    img_dir = ".\\data\\acdc\\training\\imgs"
    image_paths = os.listdir(img_dir)
    image_paths = [os.path.join(img_dir, path) for path in image_paths]
    print(image_paths)

    label_dir = ".\\data\\acdc\\training\\masks"
    label_paths = os.listdir(label_dir)
    label_paths = [os.path.join(label_dir, path) for path in label_paths]
    print(label_paths)

    # 旋转角度
    rotation_degrees = 90
    rotate_imgs(image_paths, label_paths, img_dir, label_dir, rotation_degrees)
    # adjust_brightness(image_paths)
    # enhance_contrast(image_paths)
    # add_gaussian_noise(image_paths)
