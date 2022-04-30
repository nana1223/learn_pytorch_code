import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
import time

"""
利用gpu训练方式二：
1.定义要训练的设备device = torch.device("cuda")
2.将网络模型、损失函数、训练和测试过程中的数据（输入、标签）都调用.to(device)
"""

# 定义训练的设备
device = torch.device("cuda")

# 1.准备数据集
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root="../datasets", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 2.利用 Dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 3.搭建神经网络（一般习惯把搭建的神经网络单独放入一个model.py中，然后在训练文件中引入model）
class MyNeural(nn.Module):
    def __init__(self):
        super(MyNeural, self).__init__()
        # padding和stride是根据官网给的公式计算得出
        # 使用Sequential把神经网络中的各层写到一起，简化书写
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


my_neural = MyNeural()
# 把网络转移到设备上
my_neural = my_neural.to(device)

# 4.设置损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 5.设置优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(my_neural.parameters(), lr=learning_rate)

# 6.设置训练网络的一些参数
# 记录训练的次数、测试的次数、训练的轮数
total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("../logs")

# 7.开始训练
start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始----------".format(i + 1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = my_neural(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # 这样打印可以方便观察，避免无用信息
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            # 因为loss是tensor数据类型，即Tensor(2)，而 loss.item()即为 2
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始，（每一轮训练过后，在测试数据集上跑一遍），注意在测试过程就不需要调优，不需要梯度
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = my_neural(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # argmax参数为1时横着看，参数为0时竖着看
            total_accuracy = total_accuracy + (outputs.argmax(1) == targets).sum()
    print("整体测试集上的Loss:{}".format(total_test_loss))
    # 分类问题也可以用正确率来衡量
    print("整体测试集上的正确率：{}".format(total_accuracy / total_test_step))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / total_test_step, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每轮训练的模型
    torch.save(my_neural, "my_neural{}.pth".format(i + 1))
    print("模型已保存")

writer.close()
