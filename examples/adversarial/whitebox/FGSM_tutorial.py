import numpy as np
import torch

from attacks.adversarial.whitebox.fast_gradient_method import fast_gradient_method
from models.fcn import FcnNet
from examples.adversarial.utils import load_dataset,train,accuarcy

if __name__ == '__main__':
    dataset_name = 'SyntheticControl'
    train_x, train_y, test_x, test_y = load_dataset(dataset_name)
    model = FcnNet(train_x.shape[2], torch.unique(train_y).shape[0])
    train(train_x, train_y, model)
    print(f'test acc on clean data: {accuarcy(test_x, test_y, model)}')
    adv_x = fast_gradient_method(model, test_x, 0.2, np.Inf)
    print(f'test acc on adv data: {accuarcy(adv_x, test_y, model)}')