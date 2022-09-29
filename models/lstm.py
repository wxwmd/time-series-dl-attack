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
    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()     # 先转为Tensor
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)    # DataLoader可自动生成可训练的batch数据

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

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

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        model.eval()                    # pytorch中，预测时要转换成预测模式
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, hidden_valid = model(_valid_X, hidden_valid)
            hidden_valid = None
            loss = criterion(pred_Y, _valid_Y)  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        print("The train loss is {:.6f}. ".format(train_loss_cur) +
              "The valid loss is {:.6f}.".format(valid_loss_cur))



def predict(model, config, test_X):
    # 获取测试数据
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # 加载模型
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)
    return result  # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据
