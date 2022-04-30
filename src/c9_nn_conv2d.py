import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 1.读取CIFAR10数据集，并且放在dataloader中加载
dataset = torchvision.datasets.CIFAR10("../datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 2.搭建一个简单的神经网络 my_neural（以MyNeural为骨架搭建）
class MyNeural(nn.Module):
    def __init__(self):
        super(MyNeural, self).__init__()
        # 定义一个卷积层名叫conv1
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # 把x放入卷积层conv1中
        x = self.conv1(x)
        return x


my_neural = MyNeural()

writer = SummaryWriter("../logs")
step = 0

# 3.把dataloader中加载的每张图像放到神经网络中测试一下
for data in dataloader:
    imgs, targets = data
    # output即把dataloader中的数据放入神经网络中，经过神经网络中的forward中进行一个卷积操作，然后返回的输出
    output = my_neural(imgs)

    # torch.Size([64, 3, 32, 32])
    print(imgs.shape)
    # torch.Size([64, 6, 30, 30])
    print(output.shape)

    # 用tensorboard更直观的显示一下
    writer.add_images("input", imgs, step)
    # 注意这里，卷积后输出图像的通道为6，而彩色图像为channel=3是才能显示，所以需要对输出图像处理一下，
    # 使[64, 6, 30, 30] -> [xx, 3, 30, 30]，其中batch_size不知道写多少的时候就写-1它会进行计算
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1
