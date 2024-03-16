import torch
import torch.nn as nn

conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False)
conv2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 1), padding=(1, 0), bias=False)
conv3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 3), padding=(0, 1), bias=False)

input_image = torch.randn(1, 1, 5, 5)

output1 = conv1(input_image)
output2 = conv2(input_image)
output3 = conv3(input_image)

true_output = output1 + output2 + output3

print(output1)
print(output2)
print(output3)
print(true_output)

# 获取权重数据
weights1 = conv1.weight.data
weights2 = conv2.weight.data
weights3 = conv3.weight.data

# 将conv2的权重添加到conv1的中间列，同时保持conv1其他列的权重不变
true_weights = weights1.clone()  # 先复制一份原始权重
true_weights[:, :, :, 1] += weights2.squeeze(dim=-1)
true_weights[:, :, 1, :] += weights3.squeeze(dim=2)

# 合并权重（在通道维度上对应元素相加）
# merged_weights = weights1 + weights2

merged_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False)
merged_conv.weight.data.copy_(true_weights)

predicted_output = merged_conv(input_image)
#
# print(input_image)
# print(true_output)
# print(predicted_output)
# print(torch.allclose(true_output, predicted_output, atol=1e-5))
