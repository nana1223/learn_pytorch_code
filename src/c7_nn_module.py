import torch
from torch import nn


# 自己搭建的神经网络必须从nn.Module类中继承
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # 定义了神经网络要做的前向传播的操作 （前向传播就是从输入到输出）,应该在所有的子类中都进行重写
    def forward(self, input_num):
        output_num = input_num + 1
        return output_num


# 以Model为模板创建出的一个神经滚网络neural
neural = Model()
x = torch.tensor(1.0)
# nn.Module中的__call__中调用了forward方法
output = neural(x)
print(output)
