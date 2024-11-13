import torch
import models

# old_fn = "./runs/ResNet34_ep=100_bs=48_lr=0.85_wd=0"
# epoch_num = 50
# fn = old_fn.split("_")[0] + "_ep=" + str(epoch_num) + "_" +  old_fn.split("_")[2] + "_" +  old_fn.split("_")[3] + "_" +  old_fn.split("_")[4]

# print(fn)

# state_dict = torch.load('./datasets/resnet50.pth')

# for k,v in state_dict.items():
#     print(k)

model = models.ResNet50()
params=model.state_dict() #获得模型的原始状态以及参数。
for k,v in params.items():
    print(k) #只打印key值，不打印具体参数。


