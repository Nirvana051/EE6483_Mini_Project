
from PIL import Image
from torchvision import transforms
from torch.utils import data
import os
import matplotlib.pyplot as plt
import numpy as np

class Mydata(data.Dataset):
    """定义自己的数据集"""
    def __init__(self, root, Transforms=None):
        """
        Args:
            root:训练集的路径
            Transforms:图片处理的方式
            train:是否是训练数据。True:训练数据;False:验证数据
        """
        
        """进行数据集的划分"""
        
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]  #训练集

        if Transforms is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
            # 图片处理
            color_aug = transforms.ColorJitter(brightness=0.1)
            self.transforms = transforms.Compose(
                    [ transforms.CenterCrop([224,224]), 
                    transforms.Resize([224,224]),
                    color_aug,
                    transforms.RandomHorizontalFlip(p=0.5),    #数据增强
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1),   #亮度变化
                    transforms.ToTensor(), normalize])
            
    def __getitem__(self, index):
        """
		返回一张图片的数据
		训练集和验证集,则对应的是dog返回1,猫则返回0
		"""
        img_path = self.imgs[index]

        label = 1 if 'dog' in img_path.split('/')[-1] else 0   #获得标签
        #print(img_path)
        data = Image.open(img_path).convert("RGB")  #可能有单通道的，注意转换为三通道图片
        try:
            data = self.transforms(data)   #图片处理
        except:
            print(img_path)
            raise ValueError("图片打开失败")
        return data, label
    def __len__(self):
        """返回数据集中所有图片的个数"""    
        return len(self.imgs)
    def getall(self):
        return self.imgs

if __name__ == "__main__":
    root = "./data/train"
    train = Mydata(root, train=True)
    img,label=train.__getitem__(5)
    imgs=train.getall()
    img=img.numpy()
    print(type(img))
    print(img.shape,label)
    print(len(train))
    plt.imshow(np.transpose(img, (1,2,0)))