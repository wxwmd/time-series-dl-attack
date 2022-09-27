import torch
import torch.nn.functional as F
from torch.optim import Adam

from attacks.fast_gradient_method import fast_gradient_method
from models.fcn import FcnNet
import pandas as pd
import numpy as np


DATASET_PATH = 'dataset/UCR/'

def load_dataset(dataset_name = 'SyntheticControl'):
    training_file_path = DATASET_PATH + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv'
    test_file_path = DATASET_PATH + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv'
    training_data = torch.from_numpy(pd.read_csv(training_file_path, sep='	', header=None).to_numpy())
    test_data = torch.from_numpy(pd.read_csv(test_file_path, sep='	', header=None).to_numpy())
    train_x = training_data[:, 1:].unsqueeze(1).to(torch.float32)
    train_y = training_data[:, 0].to(torch.long) - 1
    test_x = test_data[:, 1:].unsqueeze(1).to(torch.float32)
    test_y = test_data[:, 0].to(torch.long) - 1
    return train_x, train_y, test_x, test_y

def train(train_x, train_y, model):
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(100):
        pred_y = model(train_x)
        loss = F.cross_entropy(pred_y, train_y)
        print(f'epoch : {epoch}, loss: {loss.data}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def accuarcy(test_x, test_y, model):
    model.eval()
    pred_y = model(test_x)
    acc = test_y.eq(pred_y.max(dim=1)[1]).sum() / test_x.shape[0]
    return acc

if __name__ == '__main__':
    dataset_name = 'SyntheticControl'
    train_x, train_y, test_x, test_y = load_dataset(dataset_name)
    model = FcnNet(train_x.shape[2], torch.unique(train_y).shape[0])
    train(train_x, train_y, model)
    print(f'test acc on clean data: {accuarcy(test_x, test_y, model)}')
    adv_x = fast_gradient_method(model, test_x, 0.2, np.Inf)
    print(f'test acc on adv data: {accuarcy(adv_x, test_y, model)}')