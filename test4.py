import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

true_w = 5
true_h = 2.5
epoch = 10
batch_size = 10
lr = 0.01

# model = nn.Sequential(
#     nn.Linear(1, 1),
#     # nn.ReLU(),
#     # nn.BatchNorm1d(5),
#     # nn.Linear(5, 1),
# )

train_w = torch.randn(1, requires_grad=True)
train_b = torch.randn(1, requires_grad=True)
features = torch.randn(1000, 2)
features[:, 1] = features[:, 0] * true_w + true_h
features[:, 1] = features[:, 1] + torch.normal(0, 1, features[:, 1].shape)


class MyDataset(Dataset):
    def __init__(self, features):
        super(MyDataset, self).__init__()
        self.features = features

    def __getitem__(self, index):
        return self.features[index, 0], self.features[index, 1]

    def __len__(self):
        return len(self.features)


dataloader = torch.utils.data.DataLoader(MyDataset(features), batch_size=batch_size)

criterion = nn.MSELoss()

optim = torch.optim.SGD([train_w, train_b], lr=lr)

for i in range(epoch):
    epoch_loss = 0.0
    for j, data in enumerate(dataloader):
        optim.zero_grad()

        # 使用模型进行预测
        # print(feature.size())
        feature = data[0]
        label = data[1]

        output = feature * train_w + train_b

        loss = criterion(output, label)

        loss.backward()
        optim.step()

        # 更新平均损失
        epoch_loss += loss.item() * len(feature)

    avg_epoch_loss = epoch_loss / len(features)
    print(f'Epoch {i + 1}, Epoch Loss: {avg_epoch_loss:.4f}')

# 取出一些样本用于可视化
sample_features, sample_labels = features[:100, 0], features[:100, 1]

# 使用模型对样本进行预测
# predictions = model(torch.tensor(sample_features).reshape(-1, 1)).detach().numpy()
predictions = (torch.tensor(sample_features).reshape(-1, 1) * train_w + train_b).detach().numpy()

# 绘制散点图并添加预测直线
plt.scatter(sample_features, sample_labels, label='True Labels')
plt.plot(sample_features, predictions, 'r', label='Predictions')

plt.xlabel('Feature')
plt.ylabel('Label')
plt.legend()
plt.show()
