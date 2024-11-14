import os
import random
from tqdm import tqdm
import shutil


if __name__=="__main__":

    root="./data"   #数据文件根目录
    trainPath="./datasets/train"  #训练集根目录
    valPath="./datasets/val"  #验证集根目录
    train_num=20000  #训练集数目
    val_num=5000
    classes=["cat","dog"]

    for folder in os.listdir(root):   #一开始先删除里面原有的照片
        folderpath=os.path.join(root,folder)
        if os.path.isdir(folderpath):
            for pic in os.listdir(folderpath):
                if pic.endswith(".jpg"):
                    file=os.path.join(folderpath,pic)
                    os.remove(file)
        print("删除成功")

    for cls in classes:
        folder_path=os.path.join(trainPath,cls)
        pictures=os.listdir(folder_path)
        random.shuffle(pictures)  #打乱顺序
        for index in tqdm(range(int(train_num/len(classes)))):
            #复制到训练集 
            oldpath=os.path.join(folder_path,pictures[index])
            newpath=os.path.join(root,"train",str(index)+"_"+cls.lower()+".jpg")
            shutil.copy(oldpath,newpath)
    
    for cls in classes:
        folder_path=os.path.join(valPath,cls)
        pictures=os.listdir(folder_path)
        random.shuffle(pictures)  #打乱顺序
        for index in tqdm(range(int(val_num/len(classes)))):
            #复制到验证集 
            oldpath=os.path.join(folder_path,pictures[index])
            newpath=os.path.join(root,"val",str(index)+"_"+cls.lower()+".jpg")
            shutil.copy(oldpath,newpath)
        
