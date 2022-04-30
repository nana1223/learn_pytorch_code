import torch
import torchvision
from PIL import Image

# 1.准备数据
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

image_path = "../imgs/dog.png"
image = Image.open(image_path)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
image = torch.reshape(image, (1, 3, 32, 32))


# 2.加载网络模型
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


# !!! 在gpu上训练出来的模型要在cpu上测试的话需要说明一下映射
my_neural = torch.load("my_neural1.pth", map_location=torch.device('cpu'))

# 3.把image输入模型中
my_neural.eval()
with torch.no_grad():
    output = my_neural(image)
print(output)
# 第几类
print(output.argmax(1))
