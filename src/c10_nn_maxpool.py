import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

# 1.获取输入数据
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 2.构建神经网络
class MyNeural(nn.Module):
    def __init__(self):
        super(MyNeural, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


my_neural = MyNeural()

# 3.把输入数据放入含有一个池化层的神经网络中执行
writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = my_neural(imgs)
    writer.add_images("input", imgs, step)
    # 池化不会改变通道
    writer.add_images("output", output, step)
    step = step + 1
writer.close()
