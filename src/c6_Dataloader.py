import torchvision
from torch.utils.data import DataLoader

# 1.准备的测试据集
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 2.DtaLoader加载数据集
# 参数理解：dataset数据集，batch_size每次加载的数据量（把64个图片信息当成一组打包成一个作为dataloader的一个返回），
# shuffle每次加载数据之前是否重新洗牌，num_workers线程数， drop_last最后余数余下的数据集是否丢掉
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集
img, target = test_data[0]
print(img.shape)
# 输出：torch.Size([3, 32, 32]) rgb3通道，32×32大小的图
print(target)

writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(img.shape)
    # print(target)
    # 注意这里用的是add_images() 而不是add_image()
    writer.add_images("test_data", imgs, step)
    step = step + 1
writer.close()
