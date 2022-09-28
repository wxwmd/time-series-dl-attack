'''
时序回归：根据股票过去60天的数据预测当天股票的最高价
'''

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from attacks.adversarial.whitebox.fast_gradient_method import fast_gradient_method


# import data
stock_data = pd.read_csv("../../../dataset/Google stocks/Google_stocks.csv", dtype={'Close': 'float64', 'Open': 'float64', 'High': 'float64', 'Low': 'float64'})

#pick the input features (average and volume)
input_data = stock_data.iloc[:,1:].values

# data preparation
lookback = 60

train_size = int(.7 * len(stock_data))
x = []
y = []
for i in range(len(stock_data) - lookback - 1):
    t = input_data[i:i+lookback]
    x.append(t)
    y.append(input_data[i + lookback, 2])


x, y = torch.from_numpy(np.array(x)).to(torch.float32), torch.from_numpy(np.array(y)).to(torch.float32)
x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:]
y_test = y[train_size:]

print(x.shape)
print(x_test.shape)
print(y.shape)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0)) # [1431, 60, 16]
        pred = self.linear(output)  # [1431, 60, 1]
        pred = pred[:, -1, :]  # [1431, 1]
        return pred


model = LSTM(4, 16, 2, 1)
optimizer = Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
for epoch in range(1000):
    pred = model(x_train)
    loss = F.mse_loss(pred, y_train)
    print(loss.data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

pred_y_test = model(x_test)
print(f'the mse on clean test data: {F.mse_loss(pred_y_test, y_test)}')

adv_x_test = fast_gradient_method(model, x_test, eps=30, norm=np.inf)
adv_y_pred = model(adv_x_test)
print(f'the mse on clean adv data: {F.mse_loss(adv_y_pred, y_test)}')