"""
 处理ACDC数据集，转换为yolo格式的数据集
"""
import os
import shutil
import numpy as np
import cv2


def save_images2dir():
    """
    将ACDC数据集的图片作为YOLO的数据集图片，并保存到指定位置
    """
    # 遍历文件夹下的所有文件
    for filename in os.listdir(image_path):
        img_file_path = os.path.join(image_path, filename)
        yolo_img_file_path = os.path.join(yolo_image_path, filename)
        shutil.copy2(img_file_path, yolo_img_file_path)


def save_labels2dir(min_pixel, max_pixel):
    """
    将ACDC数据集的标签进行处理，作为YOLO的数据集标签，并保存到指定位置
    """
    for filename in os.listdir(label_path):
        label_file_path = os.path.join(label_path, filename)

        # 读取图像并二值化
        image = cv2.imread(label_file_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = np.zeros_like(image)
        binary_mask[(image >= min_pixel) & (image <= max_pixel)] = 1

        # 找到连通区域的边界框
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 对于单个目标或者无目标的情况
        if len(contours) == 1:
            contour = contours[0]
            x, y, w, h = cv2.boundingRect(contour)

            # 计算归一化的中心坐标和宽高
            center_x_normalized = (x + w / 2) / image.shape[1]
            center_y_normalized = (y + h / 2) / image.shape[0]
            width_normalized = w / image.shape[1]
            height_normalized = h / image.shape[0]

            yolo_label_filename = os.path.splitext(filename)[0] + '.txt'
            yolo_label_file_path = os.path.join(yolo_label_path, yolo_label_filename)

            with open(yolo_label_file_path, 'w') as f:
                f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                    center_x_normalized, center_y_normalized, width_normalized, height_normalized))
        elif len(contours) == 0:
            # 如果没有检测到任何轮廓（即没有目标）
            print(f"No target region found in the image: {filename}")
            # 可以选择创建一个空标签文件，或者不创建（取决于项目需求）
            # 若需要创建：
            # yolo_label_filename = os.path.splitext(filename)[0] + '.txt'
            # yolo_label_file_path = os.path.join(yolo_label_path, yolo_label_filename)
            # with open(yolo_label_file_path, 'w') as f:
            #     pass  # 空操作，仅创建文件而不写入内容
        else:
            print(f"Multiple target regions found in the image: {filename}, skipping...")


# 获取图像尺寸的辅助函数
def get_image_size(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return img.shape[1], img.shape[0]
    else:
        raise FileNotFoundError(f"Image not found: {image_path}")


def parse_yolo_annotation(filename, img_w, img_h):
    with open(os.path.join(yolo_label_path, filename), 'r') as file:
        lines = file.readlines()

    bboxes = []
    for line in lines:
        class_id, x, y, w, h = [float(v) for v in line.strip().split(' ')]
        if w == 0 or h == 0:
            continue
        x_center, y_center = x * img_w, y * img_h
        bbox_width, bbox_height = w * img_w, h * img_h
        x_min = max(0, round(x_center - bbox_width / 2))
        y_min = max(0, round(y_center - bbox_height / 2))
        x_max = min(img_w - 1, round(x_center + bbox_width / 2))
        y_max = min(img_h - 1, round(y_center + bbox_height / 2))
        bboxes.append((x_min, y_min, x_max, y_max, class_id))  # 添加类ID以便后续可能的颜色区分或其他用途

    return bboxes


def show_labels_on_images():
    for filename in os.listdir(yolo_label_path):
        image_file_name = filename.replace('_gt', '')
        image_file_name = image_file_name.replace('.txt', '.png')
        image_file_path = f'{yolo_image_path}\\{image_file_name}'

        # 获取图像尺寸
        img_w, img_h = get_image_size(image_file_path)
        img = cv2.imread(image_file_path)

        if img is not None:
            bboxes = parse_yolo_annotation(filename, img_w, img_h)

            # 绘制边界框
            color = (0, 255, 0)  # 示例颜色列表，根据类别ID选择不同的颜色
            for i, (x_min, y_min, x_max, y_max, class_id) in enumerate(bboxes):
                # color = colors[class_id % len(colors)]
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # 显示带有边框的图像
            cv2.imshow(f'{image_file_name}', img)
            cv2.waitKey(0)  # 按任意键继续下一个图像


def show_labels():
    for filename in os.listdir(label_path):
        label_file_path = f'{label_path}\\{filename}'
        image = cv2.imread(label_file_path, cv2.IMREAD_GRAYSCALE)

        # 将图像中像素值在1-3的部分替换成255
        image[np.where((image >= 1) & (image <= 3))] = 255

        # 这里你可以选择显示或者保存处理后的图像
        # 显示图像
        cv2.imshow('Processed Image', image)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = '..\\data\\acdc\\testing\\imgs'
    label_path = '..\\data\\acdc\\testing\\masks'
    yolo_image_path = '..\\yolo_data\\data\\images\\testing'
    yolo_label_path = '..\\yolo_data\\data\\labels\\testing'

    # save_images2dir()

    # save_labels2dir(min_pixel=1, max_pixel=3)

    show_labels_on_images()

    # show_labels()
