import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


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
print(my_neural)

# 使用torch.ones创建都是1的数，可以测试一下这个神经网络搭建的是否正确
input = torch.ones((64, 3, 32, 32))
output = my_neural(input)
print(output.shape)

writer = SummaryWriter("../logs")
# add_graph可以输出这个网络的结构图以及每层的参数大小
writer.add_graph(my_neural, input)
writer.close()
