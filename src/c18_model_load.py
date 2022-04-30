import torch
import torchvision
from c17_model_save import *

vgg16 = torchvision.models.vgg16(pretrained=False)

# 加载方式1，对应保存方式1
model = torch.load("vgg16_method1.pth")
print(model)

# 加载方式2，对应保存方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

# 陷阱1，
# 要让该.py文件加载自己定义的神经网络，需要引入自己定义的神经网络的模板类 from c17_model_save import *
model = torch.load("my_neural_method1.pth")
print(model)

