import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 加载4D nifti图像
filename = 'Resources/training/patient001/patient001_4d.nii.gz'
nii_data = nib.load(filename)
data_4d = nii_data.get_fdata()  # 获取4D数据

# 获取时间点的数量
num_time_points = data_4d.shape[3]

# 创建一个新的figure对象，用于显示多张图片
# fig, axs = plt.subplots(nrows=1, ncols=num_time_points)

# 遍历所有时间点
for t in range(16):
    # 提取当前时间点的3D数据
    slice_3d = data_4d[..., t]

    slice_2d = slice_3d[:, :, 0]
    normalized_slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
    # plt.subplot(4, 4, t + 1)
    plt.imshow(normalized_slice_2d, cmap='gray')
    plt.title(f'Image Feature {t}')
    plt.show()
    # 显示当前时间点的一个典型2D切片
    # 这里假设我们显示的是中间层面的短轴切片
    # middle_slice_index = slice_3d.shape[2] // 2
    # axs[t].imshow(slice_3d[:, :, middle_slice_index], cmap='gray')
    # axs[t].set_title(f'Time point {t}')

# 调整子图间距
# plt.tight_layout()

# 显示图形

