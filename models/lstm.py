import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Net(Module):
    '''
    pytorch预测模型，包括LSTM时序预测层和Linear回归输出层
    可以根据自己的情况增加模型结构
    '''
    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden


def train(model, config,  train_and_valid_data):
    train_X, train_Y= train_and_valid_data     # 先转为Tensor
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)    # DataLoader可自动生成可训练的batch数据

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU训练还是GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()      # 这两句是定义优化器和loss

    global_step = 0
    for epoch in range(config.epoch):
        print("Epoch {}/{}".format(epoch, config.epoch))
        model.train()                   # pytorch中，训练时要转换成训练模式
        train_loss_array = []
        hidden_train = None
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            optimizer.zero_grad()               # 训练前要将梯度信息置 0
            pred_Y, hidden_train = model(_train_X, hidden_train)    # 这里走的就是前向计算forward函数

            hidden_train = None             # 如果非连续训练，把hidden重置即可

            loss = criterion(pred_Y, _train_Y)  # 计算loss
            loss.backward()                     # 将loss反向传播
            optimizer.step()                    # 用优化器更新参数
            train_loss_array.append(loss.item())
            global_step += 1

        print("The training loss is {:.6f}.".format(np.mean(train_loss_array)))