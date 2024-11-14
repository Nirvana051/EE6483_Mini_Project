from PIL import Image
from torchvision import transforms
from torch.utils import data
import os
import matplotlib.pyplot as plt
import numpy as np

class Mydata(data.Dataset):
    """Defines a custom dataset."""
    def __init__(self, root, Transforms=None):
        """
        Args:
            root: Path to the training set.
            Transforms: Image processing transformations.
        """
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]  # Training set
        
        if Transforms is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
            # Image processing
            color_aug = transforms.ColorJitter(brightness=0.1)
            self.transforms = transforms.Compose(
                    [ transforms.CenterCrop([224,224]), 
                    transforms.Resize([224,224]),
                    color_aug,
                    transforms.RandomHorizontalFlip(p=0.5),    # Data augmentation
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1),   # Brightness variation
                    transforms.ToTensor(), normalize])
                
    def __getitem__(self, index):
        """
        Returns a single image's data.
        """
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0   # Label assignment
        data = Image.open(img_path).convert("RGB")  # Convert to RGB
        try:
            data = self.transforms(data)   # Image processing
        except:
            print(img_path)
            raise ValueError("Failed to open image")
        return data, label
    
    def __len__(self):
        """Returns the total number of images in the dataset."""    
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
