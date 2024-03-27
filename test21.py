import os

path = '..\\PyTorchDemo\\data\\acdc\\testing\\masks'

# 转换为绝对路径并确保路径正确
absolute_path = os.path.abspath(os.path.normpath(path))

for filename in os.listdir(absolute_path):
    # 只考虑.txt文件
    if filename.endswith('.png'):
        # 获取完整的文件路径
        file_path = os.path.join(path, filename)

        # 打印文件路径或执行其他操作
        # 判断文件名是否包含 '_gt' 子串
        if '_gt' in filename:
            # 构建原文件和新文件的路径
            old_filepath = os.path.join(absolute_path, filename)
            new_filename = filename.replace('_gt', '')
            new_filepath = os.path.join(absolute_path, new_filename)

            # 进行重命名操作
            os.rename(old_filepath, new_filepath)
            print(f'Renamed {old_filepath} to {new_filepath}')
