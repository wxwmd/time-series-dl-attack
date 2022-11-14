import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from attacks.adversarial.torchattack.whitebox.apgd import APGD

from models.CNN import CNN

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 1  # 训练整批数据的次数
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = False  # 表示还没有下载数据集，如果数据集下载好了就写False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 下载mnist手写数据集
train_data = torchvision.datasets.MNIST(
    root='../../../dataset/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.MNIST(
    root='../../../dataset/',
    train=False  # 表明是测试集
)

# 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
# Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # 是否打乱数据，一般都打乱
)

# 进行测试
# 为节约时间，测试时只测试前2000个
#
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_x = test_x.to(device)
# torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
test_y = test_data.test_labels[:2000]
test_y = test_y.to(device)

cnn = CNN().to(device)

# 训练
# 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差

# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

# 开始训练
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = cnn(b_x)  # 先将数据放到cnn中计算output
        loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
        optimizer.zero_grad()  # 清除之前学到的梯度的参数
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 应用梯度

        if step % 50 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss)

# 再正常测试样本上验证精确度
cnn.eval()
test_output = cnn(test_x)
pred_y = torch.max(test_output, 1)[1]
accuracy = (pred_y.data == test_y.data).sum() / test_y.shape[0]
print('test accuracy: %.2f on clean data' % accuracy)

# 使用deepfool攻击样本，验证精确度
apgd = APGD(cnn, eps=0.1)
adv_x = apgd(test_x, test_y)
adv_output = cnn(adv_x)
adv_pred_y = torch.max(adv_output, 1)[1]
adv_accuracy = (adv_pred_y.data == test_y.data).sum() / test_y.shape[0]
print('test accuracy: %.2f on adv data' % adv_accuracy)
