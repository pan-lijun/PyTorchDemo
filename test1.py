import torch

# 创建tensor
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print('tensor3', tensor1)

tensor2 = torch.zeros((2, 3, 4), dtype=torch.float32)
print('tensor2', tensor2)

tensor3 = torch.linspace(0, 100, steps=11, dtype=torch.float32)
print('tensor3', tensor3)

tensor4 = torch.normal(0, 1, (2, 3))
print('tensor4', tensor4)

tensor5 = torch.randn((2, 3))
print('tensor5:', tensor5)

tensor6 = torch.rand((2, 3))
print('tensor6:', tensor6)

tensor7 = torch.randperm(10)
print('tensor7:', tensor7)
