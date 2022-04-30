import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

"""
理解：
1. pretrained=False时，相当于使用pytorch中现有的网络模型，其中各层的参数采用默认的
2. pretrained=True时，相当于使用pytorch中现有的网络模型，但其中各层的参数采用 我们在数据集上训练好的参数
"""

# 1.使用现有的网络模型
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# 2.在现有的网络模型中添加一层
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))

# 3.修改现有网络中的某层的参数
vgg16_false.classifier[7] = nn.Linear(4096, 10)
