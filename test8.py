import os
import nibabel as nib
import matplotlib.pyplot as plt

niigz_path = './Resources/testing'
image_path = './data/acdc/testing/imgs'
label_path = './data/acdc/testing/masks'


def niigz2png():
    for patient in os.listdir(niigz_path):
        # print(patient)
        patient_path = os.path.join(niigz_path, patient)
        # print(patient_path)
        info_path = os.path.join(patient_path, 'Info.cfg')
        if os.path.exists(info_path):
            ed_value, es_value = get_ed_and_es_value(info_path)
            # print('ED:', ed_value)
            # print('ES:', es_value)
            ed_image_frame = patient + '_frame' + ed_value
            ed_image_frame_name = ed_image_frame + '.nii.gz'
            ed_label_frame = ed_image_frame + '_gt'
            ed_label_frame_name = ed_label_frame + '.nii.gz'

            es_image_frame = patient + '_frame' + es_value
            es_image_frame_name = es_image_frame + '.nii.gz'
            es_label_frame = es_image_frame + '_gt'
            es_label_frame_name = es_label_frame + '.nii.gz'

            # print(ed_image_frame_name)
            patient_ed_image_path = os.path.join(patient_path, ed_image_frame_name)
            patient_ed_label_path = os.path.join(patient_path, ed_label_frame_name)
            save_nii2png(ed_image_frame, patient_ed_image_path, is_image=True)
            save_nii2png(ed_label_frame, patient_ed_label_path, is_image=False)

            patient_es_image_path = os.path.join(patient_path, es_image_frame_name)
            patient_es_label_path = os.path.join(patient_path, es_label_frame_name)
            save_nii2png(es_image_frame, patient_es_image_path, is_image=True)
            save_nii2png(es_label_frame, patient_es_label_path, is_image=False)


def save_nii2png(frame_name, frame_path, is_image):
    if os.path.exists(frame_path):
        nifti_image_feature = nib.load(frame_path)
        data_3d_feature = nifti_image_feature.get_fdata()
        affine_feature = nifti_image_feature.affine
        slice_num = data_3d_feature.shape[2]
        for i in range(slice_num):
            slice_2d_feature = data_3d_feature[:, :, i]
            output_name = frame_name + '_slice' + f"{i + 1:02d}" + '.png'
            if is_image is True:
                output_path = os.path.join(image_path, output_name)
            else:
                output_path = os.path.join(label_path, output_name)
            plt.imsave(output_path, slice_2d_feature, cmap='gray')


def get_ed_and_es_value(info_cfg_path):
    # 打开并读取文件内容
    with open(info_cfg_path, 'r') as file:
        content = file.read()

    # 分割每行并提取ED和ES的值
    lines = content.strip().split('\n')
    ed_value = int([line.split(':')[1].strip() for line in lines if line.startswith('ED:')][0])
    es_value = int([line.split(':')[1].strip() for line in lines if line.startswith('ES:')][0])

    ed_padded = f"{ed_value:02d}"
    es_padded = f"{es_value:02d}"
    return ed_padded, es_padded


if __name__ == '__main__':
    # niigz_path = './Resources/training'
    niigz2png()
    # info_cfg_path = './Resources/training/patient001/Info.cfg'
    # get_ed_and_es_value(info_cfg_path)
