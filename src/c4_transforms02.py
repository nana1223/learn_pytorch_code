from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python __call__函数的使用：直接用类对象(call的参数) 即可调用call函数

writer = SummaryWriter("logs")

img = Image.open("../imgs/dataset/train/ants/5650366_e22b7e1065.jpg")

# 1. ToTensor 把图片数据转换成tensor数据类型
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# 2. Normalize 归一化： Normalize创建实例时要输入均值和标准差，进行归一化操作输入需要是tensor类型的图片
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

# 3.Resize 缩放图片大小
print(img.size)
# img PIL -> resize -> img_resize PIL
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor (想要在tensorboard进行显示需要再变换成tensor类型)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 1)
print(img_resize)

# 4.Compose 一个transforms的系列变化,Compose中的参数是一个列表，列表元素要是transforms类型的，即Compose([transforms1,transforms2,…])
trans_resize_2 = transforms.Resize(512)
# PIL -> resize PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 2)

"""
理解：
1.transforms.Compose相当于调用Compose类的__init__函数，把返回的实例对象赋给trans_compose
2.trans_compose(img)相当于调用Compose类的__call__函数
"""

# 5. RandomCop 随即裁剪
# 注意这里的crop size不能大于 input image size；给一个int参数时按正方形裁剪，给一个(int, int)参数时按高宽长方形裁剪
trans_random = transforms.RandomCrop(300)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
