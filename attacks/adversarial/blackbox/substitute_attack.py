'''
Practical Black-Box Attacks against Machine Learning

对于白盒的adversarial attack，因为我们可以拿到模型的梯度，可以进行梯度攻击样本。
在黑盒模式下，我们只能得到目标模型的一个接口，即我们给定输入样本，目标模型给出对这个样本的判别。

这篇文章提出的想法是：对抗样本具有可迁移性，对于目标模型，我们先训练一个替代模型去拟合其效果，
我们可以获取到这个替代模型的梯度，因此可以生成这个替代模型的对抗性样本，最终我们使用这些对抗性样本去攻击目标模型。
此攻击的示例见：examples/adversarial/blackbox/substitute_attack_tutorial.py
'''
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam

from attacks.adversarial.whitebox.fast_gradient_method import fast_gradient_method


def train_substitute_model(target_model, substitute_model, train_x):
    '''
    给定一个要攻击的黑盒目标模型(target_model)，一个替代模型(substitute_model)，训练数据集(training_x)，我们训练替代模型使其逼近目标模型的效果
    :param target_model: 黑盒目标模型，无法访问模型参数，只提供接口
    :param substitute_model: 替代模型
    :param train_x: 训练数据
    :return: substitute_model
    '''
    target_y = target_model(train_x).max(dim=1)[1]
    optimizer = Adam(substitute_model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(100):
        pred_y = substitute_model(train_x)
        loss = F.cross_entropy(pred_y, target_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return substitute_model

def substitute_attack(target_model, substitute_model, train_x, x):
    '''
    黑盒攻击
    :param target_model: 目标黑盒模型
    :param substitute_model: 替代模型
    :param train_x: 训练数据，用于训练替代模型
    :param x: 要攻击的数据
    :return: adv_x，对x进行局部扰动产生的对抗样本
    '''
    train_substitute_model(target_model, substitute_model, train_x)
    adv_x = fast_gradient_method(substitute_model,x, eps=0.2, norm=np.Inf)
    return adv_x