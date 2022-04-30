from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# 解决两个问题 ： 1.transforms该如何使用    2.为什么需要Tensor数据类型
# 理解：Tensor数据类型包装了神经网络所需要的一些参数，例如：梯度，反向传播啥的


# 1.读取图片
# 这里使用相对路径，注意绝对路径中的反斜杠在Windows下会被当作转义符
img_path = "../imgs/dataset/train/bees/21399619_3e61e5bb6f.jpg"
img = Image.open(img_path)

# 2.新建一个ToTensor对象，把图片数据转换成tensor型 （调用__call__()函数）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# 3. 使用tensorboard将训练过程的图片可视化
writer = SummaryWriter("logs")
writer.add_image("Tensor_img", tensor_img)
writer.close()

"""
常用读图片的两种方式
1.  
from PIL import Image
img_path = "dataset/train/bees/21399619_3e61e5bb6f.jpg"
img = Image.open(img_path)
这种方式的图片类型是PIL Image
2.
import cv2
cv_img = cv2.imread(img_path)
这种用opencv方式读的图片类型是numpy.ndarray
"""
