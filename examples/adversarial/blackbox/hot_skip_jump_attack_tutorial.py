import torch

from attacks.adversarial.cleverhans.blackbox.hot_skip_jump_attack import hop_skip_jump_attack
from models.fcn import FcnNet
from examples.utils import load_dataset,train,accuarcy

if __name__ == '__main__':
    dataset_name = 'SyntheticControl'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_x, train_y, test_x, test_y = load_dataset(dataset_name)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    n_classes = torch.unique(train_y).shape[0]
    model = FcnNet(train_x.shape[2], n_classes).to(device)
    train(train_x, train_y, model)
    print(f'test acc on clean data: {accuarcy(test_x, test_y, model)}')
    # 给一个攻击的target labels，让adversarial samples都北误分类为指定的label
    adv_y = ((test_y + 1) % n_classes).to(torch.long)
    adv_x = hop_skip_jump_attack(model, test_x, norm=2, y_target=None, num_iterations=64)
    print(f'test acc on adv data: {accuarcy(adv_x, test_y, model)}')