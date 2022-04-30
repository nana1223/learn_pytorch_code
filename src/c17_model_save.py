import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1，保存了网络模型的结构以及其中的参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，把网络模型的参数保存成字典，不再保存网络模型的结构（官方推荐）占的空间小
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# 陷阱，用方式1保存自己写的神经网络
class MyNeural(nn.Module):
    def __init__(self):
        super(MyNeural, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


my_neural = MyNeural()
torch.save(my_neural, "my_neural_method1.pth")
