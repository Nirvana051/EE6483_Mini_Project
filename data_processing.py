import os
import random
from tqdm import tqdm
import shutil

if __name__=="__main__":
    root="./data"   # Root data directory
    trainPath="./datasets/train"  # Training set directory
    valPath="./datasets/val"  # Validation set directory
    train_num=20000  # Number of training samples
    val_num=5000     # Number of validation samples
    classes=["cat","dog"]
    
    # Initially delete existing images
    for folder in os.listdir(root):
        folderpath=os.path.join(root,folder)
        if os.path.isdir(folderpath):
            for pic in os.listdir(folderpath):
                if pic.endswith(".jpg"):
                    file=os.path.join(folderpath,pic)
                    os.remove(file)
        print("Deletion successful")
    
    # Copy training images
    for cls in classes:
        folder_path=os.path.join(trainPath,cls)
        pictures=os.listdir(folder_path)
        random.shuffle(pictures)  # Shuffle order
        for index in tqdm(range(int(train_num/len(classes)))):
            # Copy to training set 
            oldpath=os.path.join(folder_path,pictures[index])
            newpath=os.path.join(root,"train",str(index)+"_"+cls.lower()+".jpg")
            shutil.copy(oldpath,newpath)
    
    # Copy validation images
    for cls in classes:
        folder_path=os.path.join(valPath,cls)
        pictures=os.listdir(folder_path)
        random.shuffle(pictures)  # Shuffle order
        for index in tqdm(range(int(val_num/len(classes)))):
            # Copy to validation set 
            oldpath=os.path.join(folder_path,pictures[index])
            newpath=os.path.join(root,"val",str(index)+"_"+cls.lower()+".jpg")
            shutil.copy(oldpath,newpath)
