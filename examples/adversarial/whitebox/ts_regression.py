import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from attacks.adversarial.whitebox.fast_gradient_method import fast_gradient_regression_method

from models.lstm import train, predict, Net


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

    train_data_rate = 0.95      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.15      # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 64
    learning_rate = 0.001
    epoch = 10                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42            # 随机种子，保证可复现

    # 路径参数
    train_data_path = "../../../dataset/Google stocks/stock_data.csv"


class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self.mean = np.mean(self.data, axis=0)              # 数据的均值和方差
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std   # 归一化，去量纲

        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):                # 读取初始数据
        init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()     # .columns.tolist() 是获取列名

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
                                    self.config.label_in_feature_index]    # 将延后几天的数据作为label

        # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行。。。。
        train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)   # 划分训练和验证集，并打乱
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)     # 防止time_step大于测试集数量
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行
        # 比如：1-20行，21-40行。。。到数据末尾。
        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        if return_label_data:       # 实际应用中的测试集是没有label数据的
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)


def draw(config: Config, origin_data: Data, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : ,
                                            config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]   # 通过保存的均值和方差还原数据
    assert label_data.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    # label 和 predict 是错开config.predict_day天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    print("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]

    if not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
        for i in range(label_column_num):
            plt.figure(i+1)                     # 预测数据绘制
            plt.plot(label_X, label_data[:, i], label='label')
            plt.plot(predict_X, predict_data[:, i], label='predict')
            plt.title("Predict stock {} price ".format(label_name[i]))
            print("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                  str(np.squeeze(predict_data[-config.predict_day:, i])))

        plt.show()

def main(config):
    np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
    data_gainer = Data(config)
    model = Net(config) # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中

    if config.do_train:
        train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
        train(model, config,  [train_X, train_Y, valid_X, valid_Y])

    if config.do_predict:
        test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
        pred_result = predict(model, config, test_X).detach().cpu().numpy()       # 这里输出的是未还原的归一化预测数据
        draw(config, data_gainer, pred_result)





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