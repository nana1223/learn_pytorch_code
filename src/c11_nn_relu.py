import torch

# 1.定义输入数据
from torch import nn
from torch.nn import ReLU

input_matrix = torch.tensor([[1, -0.5],
                             [-1, 3]])
# 注意这里input一定要指定一个batch_size
input_matrix = torch.reshape(input_matrix, (-1, 1, 2, 2))


# 2.定义神经网络
class MyNeural(nn.Module):
    def __init__(self):
        super(MyNeural, self).__init__()
        # ReLU()的参数inplace，若为true时，则改变后的值赋给原值；若为false，则原值不变，改变后的值赋给新变量
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output


my_neural = MyNeural()

# 3.把输入数据放入神经网络当中
output = my_neural(input_matrix)
print(output)
# 结果： tensor([[[[1., 0.],
#                 [0., 3.]]]])
