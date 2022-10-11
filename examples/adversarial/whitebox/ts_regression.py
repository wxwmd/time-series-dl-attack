import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from attacks.adversarial.whitebox.fast_gradient_method import fast_gradient_regression_method
from models.lstm import train, Net


class Config:
    # 数据参数
    feature_columns = list(range(2, 9))     # 要作为feature的列，按原数据从0开始计算，也可以用list 如 [2,4,6,8] 设置
    label_columns = [4, 5]                  # 要预测的列，按原数据从0开始计算, 如同时预测第四，五列 最低价和最高价
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)  # 因为feature不一定从0开始

    predict_day = 1             # 预测未来几天

    # 网络参数
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128           # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0.2          # dropout概率
    time_step = 20              # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它

    # 训练参数
    do_train = True
    do_predict = True
    do_adv_attack = True
    shuffle_train_data = True   # 是否对训练数据做shuffle
    use_cuda = False            # 是否使用GPU训练


    batch_size = 64
    learning_rate = 0.001
    epoch = 5                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42            # 随机种子，保证可复现
    test_data_rate = 0.2

    # 路径参数
    train_data_path = "../../../dataset/Google stocks/stock_data.csv"


class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()
        self.data_num = self.data.shape[0]
        self.mean = np.mean(self.data, axis=0)              # 数据的均值和方差
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std   # 归一化，去量纲

        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):                # 读取初始数据
        init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()     # .columns.tolist() 是获取列名

    def get_train_and_test_data(self):
        feature_data = self.norm_data[:-self.config.predict_day]
        label_data = self.norm_data[self.config.predict_day :,
                                    self.config.label_in_feature_index]    # 将延后几天的数据作为label

        # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行。。。。
        x = [feature_data[i:i+self.config.time_step] for i in range(feature_data.shape[0] - self.config.time_step)]
        y = [label_data[i:i+self.config.time_step] for i in range(feature_data.shape[0] - self.config.time_step)]

        x, y = np.array(x), np.array(y)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=self.config.test_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)   # 划分训练和验证集，并打乱
        return train_x, test_x, train_y, test_y


def main(config):
    np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
    data_gainer = Data(config)
    model = Net(config)

    train_X, test_X, train_Y, test_Y = data_gainer.get_train_and_test_data()
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
    test_X, test_Y = torch.from_numpy(test_X).float(), torch.from_numpy(test_Y).float()
    if config.do_train:
        train(model, config,  [train_X, train_Y])

    if config.do_predict:
        criterion = torch.nn.MSELoss()
        test_X.requires_grad_(True)
        test_pred, _ = model(test_X, None)
        test_loss = criterion(test_pred, test_Y)
        print(f'loss on the clean test data: {test_loss.data}')

        if config.do_adv_attack:
            adv_x = fast_gradient_regression_method(test_X, test_loss, 0.1, np.inf)
            adv_pred, _ = model(adv_x, None)
            adv_loss = criterion(adv_pred, test_Y)
            print(f'loss on the adv test data: {adv_loss.data}')


if __name__=="__main__":
    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config

    main(con)