import torch
import cv2

# 加载预训练的YOLOv5s模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = '..\\yolo_data\\data\\images\\training\\patient001_frame01_slice02.png'
image = cv2.imread(img)

# 检查模型结构
# model.print()
#
# result = model(img)
#
# result.show()
