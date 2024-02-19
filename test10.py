from PIL import Image
import numpy as np

# 打开灰度图像
img = Image.open('data/acdc/masks/patient001_frame01_gt_slice02.png').convert('L')  # 'L' 表示灰度图像模式

# 将图像转换为numpy数组以方便操作
pixels = np.array(img)

# 遍历图像的每一个像素并打印其数值
width, height = img.size
for x in range(width):
    for y in range(height):
        pixel_value = pixels[y, x]
        if pixel_value != 0:
            print(f"Pixel at ({x}, {y}) has a value of: {pixel_value}")

# 如果你想可视化这些数值，可以创建一个新的图像并填充上像素值
output_img = Image.new('L', (width, height))
output_img.putdata(pixels.flatten())
# output_img.save("pixel_values.jpg")

# 显示原始灰度图像（如果在Jupyter notebook或支持显示的环境中）
img.show()

# 注意：上述代码不会直接显示每个位置的像素数值在图像上，
# 而是在控制台打印出每个像素的数值。如果你想在图像上标注每个像素的数值，
# 则需要更复杂的图像处理和可视化技术。
