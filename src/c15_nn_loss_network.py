import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)


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

# 使用交叉熵损失函数（根据预测值和真实值的情况选匹配的损失函数？）
loss = nn.CrossEntropyLoss()

for data in dataloader:
    imgs, targets = data
    outputs = my_neural(imgs)
    result_loss = loss(outputs, targets)
    # 使用反向传播result_loss.backward()可以获取梯度
    # result_loss.backward()
    print(result_loss)
