import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

# 1.获取输入图片数据
dataset = torchvision.datasets.CIFAR10("../datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 2.搭建神经网络
class MyNeural(nn.Module):
    def __init__(self):
        super(MyNeural, self).__init__()
        # 196608 = 32 * 32 * 3 * 64
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


my_neural = MyNeural()

# 3.把输入数据放入神经网络中跑
for data in dataloader:
    imgs, targets = data
    # 把输入展平成一行
    output = torch.flatten(imgs)
    output = my_neural(output)
    print(imgs.shape)
    print(output.shape)
