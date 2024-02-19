"""
处理官网下载的ACDC到网络要求的VOC格式，按需调用
author: cvxiayixiao
Wechat: cvxiayixiao
"""
import os
from os.path import join
import nibabel as nib
import gzip
import shutil
import matplotlib.pyplot as plt
import numpy as np

ori_ACDC_train_path = './Resources/testing'


def niigz2nii():
    """
    解压每个患者的01阶段的nii.gz 和 gt.nii.gz 到输入文件夹"ACDC_nii"
    """

    input_path = ori_ACDC_train_path
    # 处理image
    # target='frame01.nii'
    # output_path = 'ACDC_nii/images'

    # 处理gt
    target = 'frame01_gt.nii'
    output_path = 'ACDC_nii/labels'

    for patient in os.listdir(input_path):
        # ACDC_challenge_20170617/training/patient001
        patient_path = join(input_path, patient)
        for niigz in os.listdir(patient_path):
            if target in niigz:
                niigzpath = join(patient_path, niigz)
                new_nii_path = join(output_path, niigz)
                shutil.copy(niigzpath, new_nii_path)


# niigz2nii()
def convert_nii_to_jpg():
    """
    将ACDC_nii/images中的nii转到VOCjpg中
    :return:
    """
    image_num = 0
    nii_path = "./ACDC_nii/labels"
    output_dir = "./VOCdevkit/VOC2007/JPEGImages"
    for patient in os.listdir(nii_path):
        patient_path = join(nii_path, patient)
        for one in os.listdir(patient_path):
            one_patient_nii_path = join(patient_path, one)
            # 加载 .nii 文件
            nii_img = nib.load(one_patient_nii_path)
            data = nii_img.get_fdata()
            # 遍历数据的每个切片，并保存为 .png 文件
            for i in range(data.shape[2]):
                image_num += 1
                # 获取当前切片数据
                slice_data = data[:, :, i]
                # 创建输出文件路径
                num = f"{image_num}".zfill(6)
                output_path = os.path.join(output_dir, f'{patient}_{num}.jpg')
                # 以灰度图像格式保存切片数据为 .png 文件
                plt.imsave(output_path, slice_data, cmap='gray')


# convert_nii_to_jpg()

def convert_nii_to_png():
    '''
    将ACDC_nii/labels中的nii转到ACDC_nii/tmp_png_label中
    此时的像素是原label 中的像素，不是网络中的分类像素，还需要一部转换
    :return:
    '''
    image_num = 0
    nii_path = "./ACDC_nii/labels"
    output_dir = "./tmp"
    for patient in os.listdir(nii_path):
        patient_path = join(nii_path, patient)
        for one in os.listdir(patient_path):
            one_patient_nii_path = join(patient_path, one)
            # 加载 .nii 文件
            nii_img = nib.load(one_patient_nii_path)
            data = nii_img.get_fdata()
            # 遍历数据的每个切片，并保存为 .png 文件
            for i in range(data.shape[2]):
                image_num += 1
                # 获取当前切片数据
                slice_data = data[:, :, i]
                # 创建输出文件路径
                num = f"{image_num}".zfill(6)
                output_path = os.path.join(output_dir, f'{patient}_{num}.png')
                # 以灰度图像格式保存切片数据为 .png 文件
                plt.imsave(output_path, slice_data, cmap='gray')


from PIL import Image


def turnto255():
    from PIL import Image
    from PIL import Image
    for i in os.listdir("tmp"):
        output_path = os.path.join("./tmp1", i)
        png_path = join("tmp", i)
        # 读取图像
        image = plt.imread(png_path)

        # 取三个通道的平均值
        im_gray = np.mean(image, axis=2)
        im_gray = Image.fromarray((im_gray * 255).astype(np.uint8)).convert("L")
        # 保存输出图像
        im_gray.save(output_path)


def rename():
    """
    处理好的VOC2007 train和test名称不同，名称不对应不能训练
    转为对应的，之前的留作副本可以对应原图
    :return:
    """
    root = "VOCdevkit/VOC2007/SegmentationClass"
    for i in os.listdir(root):
        img_old_path = join(root, i)
        new_png_path = join(root, i[-10:])
        shutil.copy(img_old_path, new_png_path)


# rename()
if __name__ == '__main__':
    # niigz2nii()
    convert_nii_to_jpg()
