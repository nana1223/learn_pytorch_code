# 注意 本例用的数据集形式是 把label放在文件夹名字上。（dataset数据集）
# 但是还有其他类型的数据集 比如说地址等类型的数据 需要对于每一个数据存放一个txt存标签

from torch.utils.data import Dataset
from PIL import Image
import os


# PIL python image library 图像处理库
# os python提供的一个os模块，包含很多操作文件和目录的函数

# 继承Dataset类
class MyData(Dataset):

    # 类的初始化函数，创建实例时运行，为class类提供全局变量
    def __init__(self, root_dir, label_dir):
        # self.xx的变量理解为类的全局变量
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, item):
        img_name = self.img_path_list[item]
        # 获取每个图片的路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 读取图片 和标签
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path_list)


# 创建类实例
root_dir = "../imgs/dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

# 把两个数据集拼接
train_dataset = ants_dataset + bees_dataset

# 测试一下 看获取的图片
img, label = ants_dataset[0]
img.show()
print(len(train_dataset))
print(len(ants_dataset))
print(len(bees_dataset))
