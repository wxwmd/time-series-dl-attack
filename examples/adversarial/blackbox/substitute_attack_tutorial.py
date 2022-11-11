import torch

from attacks.adversarial.cleverhans.blackbox.substitute_attack import substitute_attack
from examples.utils import load_dataset, train, accuarcy
from models.fcn import FcnNet

if __name__ == '__main__':
    dataset_name = 'SyntheticControl'
    train_x, train_y, test_x, test_y = load_dataset(dataset_name)
    target_model = FcnNet(train_x.shape[2], torch.unique(train_y).shape[0])
    train(train_x, train_y, target_model)
    print(f'test acc on clean data: {accuarcy(test_x, test_y, target_model)}')

    substitude_model = FcnNet(train_x.shape[2], torch.unique(train_y).shape[0])
    adv_x = substitute_attack(target_model, substitude_model, train_x, test_x)
    print(f'test acc on adv data: {accuarcy(adv_x, test_y, target_model)}')