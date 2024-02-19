# 首先确保你已经安装了pydicom库，如果没有可以使用pip进行安装：
# pip install pydicom
import os
import test15

import pydicom
import matplotlib.pyplot as plt

# 指定你的.dcm文件路径
root_path = '.\\Resources\\msd\\testing'
train_img_path = ".\\data\\msd\\testing\\imgs"
train_label_path = ".\\data\\msd\\testing\\masks"
for patient in os.listdir(root_path):
    # 病人文件夹路径
    patient_path = os.path.join(root_path, patient)
    # print(patient_path)
    files = os.listdir(patient_path)
    # 病人序号
    patient_no = patient[-2:]
    print(f"The patient{patient_no}")
    # dcm文件夹名称
    dcm_dir_name = f"P{patient_no}dicom"
    # dcm文件夹路径
    dcm_dir_path = os.path.join(patient_path, dcm_dir_name)
    # print(dcm_dir_path)
    list_txt_files = [f for f in files if f.endswith('.txt')]
    assert len(list_txt_files) == 1, "在路径中未找到或找到了多个txt文件"
    txt_file = os.path.join(patient_path, list_txt_files[0])

    with open(txt_file, 'r') as f:
        content = f.read()
        # 假设你已经读取了文件内容到变量content
        lines = content.split('\n')  # 将内容按行分割成列表
        # 去除最后一个空行
        lines = lines[:-1]
        # 行数
        line_num = len(lines)
        assert line_num % 2 == 0, "txt文件行数必须是偶数"
        # 切片数量
        slice_num = line_num // 2
        for i in range(slice_num):
            # 获取名称并移除'.\'前缀
            first_line = lines[i * 2][2:] if lines[i * 2].startswith('.\\') else lines[i * 2]
            second_line = lines[i * 2 + 1][2:] if lines[i * 2 + 1].startswith('.\\') else lines[i * 2 + 1]
            # 内轮廓标注文件路径
            icontour_path = os.path.join(root_path, first_line)
            # 外轮廓标注文件路径
            ocontour_path = os.path.join(root_path, second_line)
            # print(i)
            # print(icontour_path)
            # print(ocontour_path)
            file_name = first_line.split("\\")[-1]
            patient_num = file_name.split("-")[0]
            slice_no = file_name.split("-")[1]
            dcm_name = f"{patient_num}-{slice_no}"
            # print(dcm_name)
            dcm_path = os.path.join(dcm_dir_path, dcm_name + ".dcm")
            # print(dcm_path)
            img_path = f"{train_img_path}\\{dcm_name}.png"
            label_path = f"{train_label_path}\\{dcm_name}-label.png"
            # print(img_path)
            # print(label_path)
            test15.save_dcm2label(icontour_path, ocontour_path, dcm_path, img_path, label_path)

    # print(slice_num)
    #
    # # 获取前两行并移除'.\'前缀
    # first_line = lines[0][2:] if lines[0].startswith('.\\') else lines[0]
    # second_line = lines[1][2:] if lines[1].startswith('.\\') else lines[1]
    #
    # # 现在第一行和第二行的内容已经去除了'.\'前缀
    # print(f"第一行（去除'.\'后）: {first_line}")
    # print(f"第二行（去除'.\'后）: {second_line}")
    #
    # icontour_path = os.path.join(root_path, first_line)
    # ocontour_path = os.path.join(root_path, second_line)
    #
    # print(icontour_path)
    # print(ocontour_path)
    #
    # with open(icontour_path, 'r') as g:
    #     contents = g.read()
    #     print(contents)

# # 使用pydicom读取.dcm文件
# ds = pydicom.read_file(file_path)
#
# # 获取图像的像素数据
# image_data = ds.pixel_array
#
# # 获取一些元数据，例如患者姓名、研究日期等
# patient_name = ds.PatientName
# study_date = ds.StudyDate
#
# # 显示图像信息（可能需要根据实际图像调整）
#
#
# plt.imshow(image_data, cmap='gray')
# plt.show()
