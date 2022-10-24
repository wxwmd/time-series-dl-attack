import torch
import numpy as np

from attacks.adversarial.whitebox.fast_gradient_method import fast_gradient_method
from attacks.adversarial.whitebox.projected_gradient_descent import projected_gradient_descent

from examples.adversarial.utils import load_dataset, train, accuarcy
from models.fcn import FcnNet

if __name__ == '__main__':
    dataset_name = 'SyntheticControl'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_x, train_y, test_x, test_y = load_dataset(dataset_name)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    model = FcnNet(train_x.shape[2], torch.unique(train_y).shape[0]).to(device)
    train(train_x, train_y, model)
    print(f'test acc on clean data: {accuarcy(test_x, test_y, model)}')
    adv_x = projected_gradient_descent(model, test_x, eps=0.1, eps_iter=0.01, nb_iter=100, norm=np.inf)
    print(f'test acc on adv data: {accuarcy(adv_x, test_y, model)}')