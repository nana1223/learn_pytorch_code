import torch
import torch.nn.functional as F

# 1.定义输入图像
input_matrix = torch.tensor([[1, 2, 0, 3, 1],
                             [0, 1, 2, 3, 1],
                             [1, 2, 1, 0, 0],
                             [5, 2, 3, 1, 1],
                             [2, 1, 0, 1, 1]])

# 2.定义卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 3.使用pytorch提供的尺寸变换，使输入数据满足卷积函数的输入(N, C, H, W) or (C, H, W)，
# 其中N就是batch_size也就是输入图片的数量，C就是通道数而这里的只是一个二维张量所以通道为1，H就是高，W宽，所以是1155
input_matrix = torch.reshape(input_matrix, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input_matrix.shape)

# 4.使用卷积函数
# conv2d：对由多个输入平面组成的输入信号进行二维卷积
output_matrix = F.conv2d(input_matrix, kernel, stride=1)
print(output_matrix)

# stride为卷积操作每次移动的步径大小
output2 = F.conv2d(input_matrix, kernel, stride=2)
print(output2)

# padding:在输入图像的边缘是否进行填充（使卷积核能作用于输入图像的每一个像素点），填充后的空白处默认为0
output3 = F.conv2d(input_matrix, kernel, stride=1, padding=1)
print(output3)
