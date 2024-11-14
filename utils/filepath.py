
import os

def find_lastfolder(model_name):
    """返回最后一次训练的文件夹"""
    path="./runs"
    
    all=os.listdir(path)
    folders=[]
    for folder in all:
        if model_name in folder:
            folders.append(folder)
    if len(folders)==0:
        return 0
    if len(folders)==1:
        return path+"/"+model_name
    folders=sorted(folders,key=lambda x:x.split("_")[-1])
    num=int(folders[-1].split("_")[-1])
    return path+"/"+model_name+"_"+str(num)

def create_newfolder(model_name):
    """创建新文件夹"""
    path="./runs"
    
    all=os.listdir(path)
    folders=[]
    for folder in all:
        if model_name in folder:
            folders.append(folder)
    if len(folders)==0:
        return path+"/"+model_name
    if len(folders)>=1:
        return path+"/"+model_name+"_"+str(len(folders))
    

if __name__=="__main__":
    a=find_lastfolder("LeNet")
    print(a)