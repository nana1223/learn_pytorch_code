import torchvision

# CIFAR10数据集包括60000张32×32的彩色图片，属于10个类型
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 下载CIFAR10数据集的压缩文件及解压，存储到./dataset路径（分别下训练数据集和测试数据集），并且对数据集中的所有图片做transform转换
train_set = torchvision.datasets.CIFAR10(root="./datasets", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=dataset_transform, download=True)

# 测试输出一下数据集信息
# print(test_set.classes)
# print(test_set[0])
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# 放在tensorboard里展示一下
writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
writer.close()
