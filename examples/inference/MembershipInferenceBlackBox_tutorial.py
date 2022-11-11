import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

from examples.utils import load_dataset_numpy, accuarcy_art
from models.fcn import FcnNet

# 训练好一个模型，这就是我们要攻击的对象
dataset_name = 'ElectricDevices'
num_classes = 7
train_x, train_y, test_x, test_y = load_dataset_numpy(dataset_name)
model = FcnNet(train_x.shape[2], num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 96),
    nb_classes=num_classes,
)
classifier.fit(train_x, train_y, batch_size=64, nb_epochs=100)
print(f'test acc on clean data: {accuarcy_art(test_x, test_y, classifier)}')


attack = MembershipInferenceBlackBox(classifier, input_type="loss", attack_model_type="nn")
# 训练数据和测试数据抽出7000条训练攻击器
attack.fit(train_x[0:7000], train_y[0:7000], test_x[0:7000], test_y[0:7000])


infer_p = attack.infer(train_x[7000:], train_y[7000:])
infer_n = attack.infer(test_x[7000:], test_y[7000:])

print(f'predicted possibility of training data: {infer_p.mean()}, predicted possibility of non-training data: {infer_n.mean()}')