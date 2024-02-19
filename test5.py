import nibabel as nib
import matplotlib.pyplot as plt

# 加载 nii.gz 文件
feature_path = 'Resources/training/patient001/patient001_frame01.nii.gz'
label_path = 'Resources/training/patient001/patient001_frame01_gt.nii.gz'
nifti_image_feature = nib.load(feature_path)
nifti_image_label = nib.load(label_path)

# 获取3D体数据和头部（affine矩阵）
data_3d_feature = nifti_image_feature.get_fdata()
affine_feature = nifti_image_feature.affine
data_3d_label = nifti_image_label.get_fdata()
affine_label = nifti_image_label.affine

# 选择要显示的某一个切片，例如第10个切片
slice_num = 9
slice_2d_feature = data_3d_feature[:, :, slice_num]
slice_2d_label = data_3d_label[:, :, slice_num]

# 将数据归一化以便于显示
normalized_slice_feature = (slice_2d_feature - slice_2d_feature.min()) / (
        slice_2d_feature.max() - slice_2d_feature.min())
normalized_slice_label = (slice_2d_label - slice_2d_label.min()) / (slice_2d_label.max() - slice_2d_label.min())

# 显示该切片
# plt.imshow(normalized_slice_feature, cmap='gray', origin='lower')
# plt.title(f'Slice #{slice_num}')
# plt.show()

plt.subplot(1, 2, 1)
plt.imshow(slice_2d_feature, cmap='gray')
plt.title('Image Feature')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(slice_2d_label, cmap='gray')
plt.title('Image Label')
plt.axis('off')

plt.tight_layout()  # 调整子图间距
plt.axis('off')
plt.show()
