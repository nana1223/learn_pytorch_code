import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

outputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

outputs = torch.reshape(outputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# 1.使用损失函数L1Loss：预测值和真实值直接相减，然后求和或者求均值 （这里使用求和方式）
loss = L1Loss(reduction='sum')
result_l1 = loss(outputs, targets)

# 2.使用损失函数MSELoss：相减后求平方，然后求和或者求均值 （默认是求均值）
loss_mse = MSELoss()
result_mse = loss_mse(outputs, targets)

print(result_l1)
# 输出：2 （1-1 + 2-2 + 5-3 = 2）
print(result_mse)
# 输出：1.333 （1-1^2 + 2-2^2 + (5-3)^2)/3 = 4/3 = 1.333

# 3.使用交叉熵损失函数CrossEntropyLoss
outputs = torch.tensor([0.1, 0.2, 0.3])
targets = torch.tensor([1])
outputs = torch.reshape(outputs, (1, 3))
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(outputs, targets)
print(result_cross)
