import os

image_path = '..\\PyTorchDemo\\data\\acdc\\validating\\imgs'
mask_path = '..\\PyTorchDemo\\data\\acdc\\validating\\masks'
label_path = '..\\PyTorchDemo\\data\\acdc\\validating\\labels'

for file_fullname in os.listdir(image_path):
    filename = file_fullname.split('.')[0]
    label_name = os.path.join(label_path, filename + '.txt')
    if os.path.exists(label_name):
        continue

    image = os.path.join(image_path, file_fullname)
    mask = os.path.join(mask_path, file_fullname)

    os.remove(image)
    os.remove(mask)
