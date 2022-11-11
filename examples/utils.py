import pandas as pd
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import random

DATASET_PATH = '../../dataset/UCR'

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

def load_dataset_numpy(dataset_name = 'SyntheticControl'):
    training_file_path = DATASET_PATH + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv'
    test_file_path = DATASET_PATH + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv'
    training_data = torch.from_numpy(pd.read_csv(training_file_path, sep='	', header=None).to_numpy())
    test_data = torch.from_numpy(pd.read_csv(test_file_path, sep='	', header=None).to_numpy())
    train_x = training_data[:, 1:].unsqueeze(1).to(torch.float32)
    train_y = training_data[:, 0].to(torch.long) - 1
    test_x = test_data[:, 1:].unsqueeze(1).to(torch.float32)
    test_y = test_data[:, 0].to(torch.long) - 1
    train_x = train_x.numpy()
    train_y = train_y.numpy()
    test_x = test_x.numpy()
    test_y = test_y.numpy()
    return train_x, train_y, test_x, test_y

def train(train_x, train_y, model, num_epochs=100):
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(num_epochs):
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

def accuarcy_art(test_x, test_y, classifier):
    pred_y = classifier.predict(test_x)
    acc = np.sum(np.argmax(pred_y, axis=1) == test_y) / len(test_y)
    return acc