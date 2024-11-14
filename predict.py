import os
import torch
from torchvision import transforms
from PIL import Image
import csv

import models

def predict(root, imgname, model, img_trans):
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
    
    pred_str = "cat" if pred_idx == 0 else "dog"
    id = imgname.split("_")[0]
    label = imgname.split("_")[1].split(".")[0]
    filepath = "./output/submission.csv"
    csv_writer(filepath, id, label, pred_str)
    return label, pred_str

def csv_writer(filepath, id, label, pr):
    with open(filepath, "a", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow([id, label, pr])

if __name__ == "__main__":
    # Image preprocessing
    img_trans = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = models.ResNet34() # Model structure
    modelpath = "./runs/ResNet34_ep=100_bs=32_lr=5e-05_ld=0.9_wd=0/ResNet34_best.pth" # Trained model path
    checkpoint = torch.load(modelpath)  
    model.load_state_dict(checkpoint)  # Load model parameters
    
    with open("./output/submission.csv", "w", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label", "predict result"])
    
    root = "./datasets/test"
    
    all = 0
    correct = 0
    
    pathlist = os.listdir(root)
    pathlist = sorted(pathlist, key=lambda x: int(x.split("_")[0]))
    
    for pic in pathlist:
        all += 1
        if pic.endswith(".jpg"):
            label, pre = predict(root, pic, model, img_trans)
            if label == pre:
                correct += 1
    print(f"Accuracy = {correct/all}, Total = {all}, Correct = {correct}")
