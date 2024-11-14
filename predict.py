
import os
import torch
from torchvision import transforms
from PIL import Image
# import matplotlib.pyplot as plt
import models
import csv

def predict(root,imgname, model, img_trans):
    imgpath = os.path.join(root, imgname)
    img_rgb = Image.open(imgpath).convert("RGB")
    img_tensor = img_trans(img_rgb).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred_prob, pred_idx = torch.max(prob, dim=1)
        pred_prob = pred_prob.item()
        pred_idx = pred_idx.item()

    if pred_idx == 0:
        pred_str = "cat"
    else:
        pred_str = "dog"
    id = imgname.split("_")[0]
    label = imgname.split("_")[1]
    label = label.split(".")[0]
    filepath = "./output/submission.csv"
    csv_writer(filepath, id, label, pred_str)
    return label, pred_str

    # plt.imshow(img_rgb)
    # plt.title("Predicted: {} ,Probability: {:.2f}".format(pred_str, pred_prob))
    # plt.savefig("output/pre_" + imgname)

def csv_writer(filepath, id, label, pr):
    with open(filepath,"a", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        #写入多行用writerows
        writer.writerow([id,label,pr])


if __name__ == "__main__":

    # 图像预处理
    img_trans = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = models.ResNet34() # 模型结构
    # modelpath = "./runs/SqueezeNet_ep=100_bs=32_lr=5e-05_ld=0.9_wd=0/SqueezeNet_best.pth"
    modelpath = "./runs/ResNet34_ep=100_bs=32_lr=5e-05_ld=0.9_wd=0/ResNet34_best.pth" # 训练好的模型路径
    checkpoint = torch.load(modelpath)  
    model.load_state_dict(checkpoint)  # 加载模型参数
    
    with open("./output/submission.csv","w", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        writer.writerow(["id","label","predict result"])

    root = "./datasets/test"

    all = 0
    correct = 0

    pathlist = os.listdir(root)
    pathlist = sorted(pathlist,key=lambda x: int(x.split("_")[0]))

    for pic in pathlist:
        all = all + 1
        if pic.endswith(".jpg"):
            label,pre = predict(root,pic, model, img_trans)
            if label == pre:
                correct = correct + 1
    print("acc = " + str(correct/all) + " all = " + str(all) + " correct = " + str(correct))
