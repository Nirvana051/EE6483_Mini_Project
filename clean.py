import os
from PIL import Image
import imghdr
from tqdm import tqdm

def is_valid_image(path):
    """Determines if an image is valid."""
    try:
        bValid = True
        fileObj = open(path, 'rb') # Open in binary mode
        buf = fileObj.read()
        
        # Method 1: Check if it starts with \xff\xd8
        if not buf.startswith(b'\xff\xd8'): 
            bValid = False
        # Method 2: Check if it ends with \xff\xd9
        elif buf[6:10] in (b'JFIF', b'Exif'):
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'): 
                bValid = False
        # Method 3: Check file type
        elif imghdr.what(path) is None:
            bValid=False
        # Method 4: Use Image.verify()
        else:
            try:
                Image.open(fileObj).verify()   # Raises error if corrupted
            except Exception as e:
                bValid = False
                print(e)
    except Exception as e:
        return False
    return bValid

if __name__=="__main__":
    root="datasets/train"  # Directory path
    Classify=["Cat","Dog"]
    del_nums={}   # Deleted image counts
    normal_nums={}  # Valid image counts
    for _cls in Classify:
        file_dir=os.path.join(root,_cls)  # Class directory path
        # Iterate through images in class directory
        for file in tqdm(os.listdir(file_dir)):
            filepath=os.path.join(file_dir,file)   # Full file path
            if is_valid_image(filepath) is False:
                # Store count of deleted images per class
                if del_nums.get(_cls,0)==0:
                    del_nums[_cls]=1
                else:
                    del_nums[_cls]+=1
                os.remove(filepath)  # Delete corrupted image
            else:
                if normal_nums.get(_cls,0)==0:
                    normal_nums[_cls]=1
                else:
                    normal_nums[_cls]+=1
    for Cls,_ in del_nums.items():
        print(f"{Cls} class: Deleted {del_nums[Cls]} images, Remaining {normal_nums[Cls]} images")
