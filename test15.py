import matplotlib.pyplot as plt
import pydicom
from matplotlib.patches import Polygon
from skimage.draw import polygon
import numpy as np
from PIL import Image


def save_dcm2png(icontour_path, ocontour_path, dcm_path, img_path, label_path):
    # 加载DICOM图像
    dcm_file = pydicom.dcmread(dcm_path)
    img_array = dcm_file.pixel_array  # 获取像素数据
    width, height = img_array.shape[1], img_array.shape[0]
    fig, ax = plt.subplots()
    ax.imshow(img_array, cmap=plt.cm.gray)  # DICOM图像通常为灰度图像
    with open(icontour_path, 'r') as file:
        lines = file.readlines()
    positions = [[float(coord) for coord in line.strip().split(' ')] for line in lines]
    # print(positions)
    # 提取x和y坐标，并将y坐标转换以适应matplotlib坐标系
    x_coords1, y_coords1 = zip(*positions)
    x_coords1 = [int(x) for x in x_coords1]
    y_coords1 = [(height - int(y)) for y in y_coords1]
    # 转换坐标以适应matplotlib坐标系统（这里假设DICOM图像中的坐标与图像像素坐标一致）
    # 注意：如果实际坐标需要根据特定的DICOM元数据进行调整，请查阅dcm_file的相关属性
    x_coords1 = [int(x) for x in x_coords1]
    y_coords1 = [height - int(y) for y in y_coords1]  # 注意y轴方向可能需要翻转
    # # 连接点形成闭合轮廓
    # ax.plot(x_coords1, y_coords1, 'r-', lw=2, transform=ax.transData)  # 使用transData在原始数据坐标系上绘制
    # ax.autoscale_view()  # 更新视图范围
    with open(ocontour_path, 'r') as file:
        lines = file.readlines()
    positions = [[float(coord) for coord in line.strip().split(' ')] for line in lines]
    # print(positions)
    # 提取x和y坐标，并将y坐标转换以适应matplotlib坐标系
    x_coords2, y_coords2 = zip(*positions)
    x_coords2 = [int(x) for x in x_coords2]
    y_coords2 = [(height - int(y)) for y in y_coords2]
    # 转换坐标以适应matplotlib坐标系统（这里假设DICOM图像中的坐标与图像像素坐标一致）
    # 注意：如果实际坐标需要根据特定的DICOM元数据进行调整，请查阅dcm_file的相关属性
    x_coords2 = [int(x) for x in x_coords2]
    y_coords2 = [height - int(y) for y in y_coords2]  # 注意y轴方向可能需要翻转
    # # 连接点形成闭合轮廓
    # ax.plot(x_coords2, y_coords2, 'b-', lw=2, transform=ax.transData)  # 使用transData在原始数据坐标系上绘制
    # ax.autoscale_view()  # 更新视图范围
    # 创建一个Polygon对象表示圆环区域
    poly_points = list(zip(x_coords1 + x_coords2[::-1], y_coords1 + y_coords2[::-1]))
    polygon = Polygon(poly_points, facecolor='g', edgecolor='none')
    # 将Polygon添加到图像中
    ax.add_patch(polygon)
    plt.savefig(label_path)
    plt.show()


def save_dcm2label(icontour_path, ocontour_path, dcm_path, img_path, label_path):
    # 加载DICOM图像并转换为numpy数组
    dcm_file = pydicom.dcmread(dcm_path)
    img_array = dcm_file.pixel_array.astype(np.uint16)

    # 保存为PNG格式的灰度图像
    # np.savetxt('label_data.txt', img_array, fmt='%d')
    plt.imsave(img_path, img_array, cmap='gray')

    # 获取图像尺寸
    width, height = img_array.shape

    # 读取内、外轮廓点坐标，并转换为整数坐标
    with open(icontour_path, 'r') as file1:
        lines1 = file1.readlines()
    positions1 = [[int(float(coord)) for coord in line.strip().split(' ')] for line in lines1]
    x_coords1, y_coords1 = zip(*positions1)

    with open(ocontour_path, 'r') as file2:
        lines2 = file2.readlines()
    positions2 = [[int(float(coord)) for coord in line.strip().split(' ')] for line in lines2]
    x_coords2, y_coords2 = zip(*positions2)

    # 创建一个与图像同样大小的空白二值标签图（默认填充0）
    label_array = np.zeros(img_array.shape, dtype=np.uint8)

    # 先将外部轮廓内的像素设置为1
    if len(positions2):
        rr_outer, cc_outer = polygon(x_coords2, y_coords2)
        label_array[cc_outer, rr_outer] = 1

    # 然后将内部轮廓内的像素重新设置为0
    rr_inner, cc_inner = polygon(x_coords1, y_coords1)
    label_array[cc_inner, rr_inner] = 0

    # 最终，标签图上只有内外轮廓之间的部分被标记为1，其它部分为0

    # # 使用imshow显示该二值图像
    # plt.imshow(label_array, cmap='gray', vmin=0, vmax=1)
    # plt.colorbar(ticks=[0, 1], label='Label')
    # plt.show()

    # np.savetxt('label_data.txt', label_array, fmt='%d')

    # 将二值数组直接作为灰度图像保存
    plt.imsave(label_path, label_array, cmap='gray', vmin=0, vmax=1)


if __name__ == '__main__':
    icontour_path = '.\\TrainingSet\\patient01\\P01contours-manual\\P01-0100-icontour-manual.txt'
    ocontour_path = '.\\TrainingSet\\patient01\\P01contours-manual\\P01-0100-ocontour-manual.txt'
    dcm_path = '.\\TrainingSet\\patient01\\P01dicom\\P01-0100.dcm'
    img_path = '.\\data\\msd\\training\\imgs\\P01-0100.png'
    label_path = '.\\data\\msd\\training\\tmp\\P01-0100-label.png'
    save_dcm2label(icontour_path, ocontour_path, dcm_path, img_path, label_path)
    # save_dcm2png(icontour_path, ocontour_path, dcm_path, img_path, label_path)
