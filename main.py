import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm, trange
from torchvision import transforms
import time
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

from utils.dataset import Mydata
import models
from utils import filepath

def train(trainPath, valPath, epoch_num, model_name, batch_size, lr, lr_decay, wd, resume, pre, cifar, op):
    """Loads data, initializes model, and trains."""
    if cifar:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, transform=transform_train, download=True)
        validset = torchvision.datasets.CIFAR10(root='./datasets', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    else:
        trainset = Mydata(trainPath)
        validset = Mydata(valPath)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU availability
    
    if pre:  
        # Transfer learning model
        model = getattr(models, model_name)()
        modelpath = "./datasets/ResNet34.pth" # Trained model path
        checkpoint = torch.load(modelpath) 
        model.load_state_dict(checkpoint, strict=False)  # Load model parameters
        
        for parma in model.parameters():
            parma.requires_grad = False  # Freeze pretrained weights
        
        class_num = 10 if cifar else 2
        
        if model_name == "ResNet50":
            model.fc = torch.nn.Linear(2048, class_num)
        elif model_name =="ResNet34":
            model.fc = torch.nn.Linear(512, class_num)
        
        model = model.cuda()
        
        criterion = torch.nn.CrossEntropyLoss()
        if op == "Adam":
            optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)
        elif op == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=wd)
        scaler = torch.amp.GradScaler()
        start_epoch = 0
        save_foldername = f"{model_name}_pretrain_ep={epoch_num}_bs={batch_size}_lr={lr}_ld={lr_decay}_wd={wd}"
        if cifar:
            save_foldername += "_cifar"
        save_foldername = filepath.create_newfolder(save_foldername)    
        writer = SummaryWriter(save_foldername)   # TensorBoard writer
    else:
        if resume:	# Resume training and load pretrained weights
            save_foldername = f"{model_name}_ep={epoch_num}_bs={batch_size}_lr={lr}_ld={lr_decay}_wd={wd}"
            if cifar:
                save_foldername += "_cifar"
            save_foldername = filepath.find_lastfolder(save_foldername)   # Retrieve last folder path
            model_path = os.path.join(save_foldername, f"{model_name}_last.pth")
            checkpoint = torch.load(model_path)	# Load checkpoint
            model = getattr(models, model_name)() 
            model.load_state_dict(checkpoint['model'])	# Load weights
            optimizer = checkpoint['optimizer']	# Load optimizer
            lr = checkpoint["lr"]
            start_epoch = checkpoint['epoch']	# Load epoch
        
            criterion = nn.CrossEntropyLoss().to(device)
            writer = SummaryWriter(save_foldername)
            scaler = torch.amp.GradScaler()
            print(f'Loaded epoch {start_epoch} successfully!')      
        else:
            model = getattr(models, model_name)()         
            if op == "Adam":
                optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)
            elif op == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=wd)
            criterion = nn.CrossEntropyLoss().to(device)
            start_epoch = 0
            # Save folder path
            filename = f"{model_name}_ep={epoch_num}_bs={batch_size}_lr={lr}_ld={lr_decay}_wd={wd}"
            if cifar:
                filename += "_cifar"
            save_foldername = filepath.create_newfolder(filename)    
            writer = SummaryWriter(save_foldername)   # TensorBoard writer
            scaler = torch.amp.GradScaler()
            print("Created new folder, starting training from scratch...")
    
    # Start training
    print('Start training...')
    model = model.cuda()
    best_accuracy = 0
    previous_loss = 1e10
    for epoch in range(start_epoch, epoch_num):
        train_loss = 0
        train_total = 0
        correct = 0
        total = 0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()   # Zero gradients
            output = model(data)    # Forward pass
            loss = criterion(output, target)   # Compute loss
            train_total += 1
            train_loss += loss.item()
            scaler.scale(loss).backward()  # Backward pass with AMP
            scaler.step(optimizer)
            scaler.update()
            _, predicted = torch.max(output.data, 1)   # Predictions
            total += target.size(0)
            correct += (predicted == target).sum()  # Correct predictions
    
        # Validation
        Loss, accuracy = validation(model, test_loader, device, criterion)  # Validation
        train_Loss = train_loss / train_total   # Training loss
        train_accuracy = correct / total  # Training accuracy
        
        # TensorBoard visualization
        writer.add_scalar('LearnRate', lr, epoch)
        writer.add_scalar('TrainLoss', train_Loss, epoch)
        writer.add_scalar('val_Loss', Loss, epoch)
        writer.add_scalar('val_accuracy', accuracy, epoch)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        print(f"Epoch:{epoch+1}/{epoch_num}, Trn_Loss:{train_Loss}, Val_Loss:{Loss}, Val_Acc:{accuracy}%, Trn_Acc:{train_accuracy}")
        
        if accuracy > best_accuracy:  # Save the best model
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(save_foldername, f"{model_name}_best.pth"))
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer,
            'epoch': epoch,
            "lr": lr
        }
        torch.save(checkpoint, os.path.join(save_foldername, f"{model_name}_last.pth"))  # Save checkpoint
        
        # Adjust learning rate if validation loss does not decrease
        if Loss >= previous_loss:
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        previous_loss = Loss
        print("Current Learning Rate:", lr)
    
    writer.close()    

@torch.no_grad()
def validation(model, test_loader, device, criterion):
    """Validation module during training."""
    model.eval() # Set model to evaluation mode
    total = 0
    correct = 0
    test_loss = 0
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)   # Forward pass
        _, predicted = torch.max(outputs.data, 1)   # Predictions
        total += labels.size(0)
        correct += (predicted == labels).sum()  # Correct predictions
        loss1 = criterion(outputs, labels)
        test_loss += loss1.item()
    model.train() # Set model back to training mode
    
    accuracy = np.around((100 * correct.cpu() / total).numpy(), decimals=2)
    Loss = round(test_loss / total, 6)
    
    return Loss, accuracy

if __name__=="__main__":
    trainPath="./data/train"    # Training set path
    valPath = "./data/val"      # Validation set path
    epoch_num = 100             # Number of epochs
    modelname = "ResNet34"      # Model name
    batchsize = 32              # Batch size
    resume = False              # Whether to resume training
    lr = 5e-5                   # Learning rate
    lr_decay = 0.9              # Learning rate decay factor
    weight_dacay = 5e-4         # Weight decay
    pre = False                 # Whether to use a pre-trained model
    cifar = False               # Whether using CIFAR-10
    op = "Adam"                 # Optimizer choice
    
    # Start training
    start = time.time()
    train(trainPath, valPath, epoch_num, modelname, batchsize, lr, lr_decay, weight_dacay, resume, pre, cifar, op)
    end = time.time()
    print(f"Training Time: {round((end - start) / 60, 2)} minutes")
