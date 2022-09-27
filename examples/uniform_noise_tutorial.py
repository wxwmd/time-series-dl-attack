import numpy as np
import torch

from attacks.fast_gradient_method import fast_gradient_method
from models.fcn import FcnNet
from models.resnet import ResNet1D
from examples.utils import load_dataset,train,accuarcy

if __name__ == '__main__':
    dataset_name = 'SyntheticControl'
    train_x, train_y, test_x, test_y = load_dataset(dataset_name)
    model = ResNet1D(in_channels=1, base_filters=5, kernel_size=9, stride=1, groups=1, n_block=10, n_classes=torch.unique(train_y).shape[0])
    train(train_x, train_y, model, num_epochs=200)
    print(f'test acc on clean data: {accuarcy(test_x, test_y, model)}')
    adv_x = fast_gradient_method(model, test_x, 0.2, np.Inf)
    print(f'test acc on adv data: {accuarcy(adv_x, test_y, model)}')