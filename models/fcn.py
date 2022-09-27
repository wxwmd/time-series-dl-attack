import torch
import torch.nn as nn
from torch.nn import Conv1d,Linear
import torch.nn.functional as F

class FcnNet(nn.Module):
    def __init__(self, time_length, num_classes):
        super().__init__()
        self.layer1 = Conv1d(1, 64, 9, padding=9//2)
        self.layer2 = Conv1d(64,32, 9, padding=9//2)
        self.layer3 = Conv1d(32, 1, 9, padding=9//2)
        self.mlp = Linear(time_length, num_classes)

    # 输入：[N:数据条数, Channel:通道数, L:每个通道上的特征]
    # 输出：[N:数据条数, c:预测的类别]
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = x.squeeze(1)
        x = self.mlp(x)
        x = F.log_softmax(x, dim=1)
        return x