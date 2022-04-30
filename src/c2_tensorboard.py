# from torch.utils.tensorboard import SummaryWriter
#
# # 创建SummaryWriter类的实例，并且将生成的文件放在名为logs的文件夹下(logs在该项目文件夹下）
# writer = SummaryWriter("logs")
#
# for i in range(100):
#     # add_scalar函数生成需要的曲线图，将生成的图存于上面函数创建的logs文件夹下，其中函数的第一个参数是要生成的图的名字，第二的参数是y轴，第三个参数是x轴
#     writer.add_scalar("y=x", i, i)
#
# # 关闭该变量
# writer.close()
#
# # 此时生成的曲线图信息我们是无法直接打开的，需要打开命令行，或者使用pycharm中的命令行，输入tensorboard --logdir=生成图所在路径。
# # 如果是在pycharm命令行打开，那么我们的生成图所在路径直接就是logs，即：tensorboard --logdir=logs；如果是其他目录下则要将完整路径写出。





# 2.add_image()函数
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 先通过SummaryWriter函数获得该变量writer，并且将生成的文件放在名为logs的文件夹下
writer = SummaryWriter("logs")
img_path = "../imgs/data/train/ants_image/5650366_e22b7e1065.jpg"
# 通过调用PIL包中的Image.open的函数获取到PIL类型的图片数据，但因add_image函数中的图片参数为numpy型或是torch.Tensor型，因此通过numpy.array函数转化图片类型
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
# add_image函数的参数分别代表：第一个参数为输出的图表名；第二个参数为图数据；第三个参数为训练的步骤；即第几步，第四个参数为第二个参数输入的格式是高、宽、通道数
writer.add_image("test", img_array, 1, dataformats="HWC")

# 运行完后，命令行输入tensorboard --logdir=logs，即可在tensorboardUI中看到训练中加的图片啥的可视化信息
