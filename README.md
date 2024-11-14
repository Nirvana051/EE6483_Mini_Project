# README

## Problem description

Based on the data of the training set, a model is trained to predict the probability that the animal in the unknown picture is a dog or a cat by using the well-trained model.

The training set consists of 20,000 pictures, and the test set contains 5,000 pictures. 

The outline is as follows：
![image](https://github.com/user-attachments/assets/7b5444ff-1728-49a9-9d6a-16e455a5ea4f)


## Datasets processing

### 1.Damaged picture cleaning

In the `clean.py` file, there are several ways to clean damaged images: 

1. Determine if the file starts with JFIF.
2. Use the imghdr library's imghdr.what function to determine the file type.
3. Use the Image.open(filename).verify() method to verify if the image is damaged.

### 2.The images are extracted to form a data set

You can copy **any given number of pictures** from the original picture folder to the **train** folder and rename it as follows:

The program is: 'data_processing.py'.

## Image preprocessing

The image preprocessing part needs to be completed:

1. Crop the picture: Crop the picture of different sizes into the required neural network, I chose to crop it as **(224x224)**
2. Convert to tensors
3. Normalization: normalization in three directions
4. Image data enhancement
5. Form the loader: Return the image data and corresponding labels, using the Pytorch Dataset package

## Models

The models are placed in 'models.py' and use some classic CNN models:

1. LeNet
2. AlexNet
3. ResNet
4. SqueezeNet

## Train

Training in 'main.py' is mainly the integration of data acquisition, training, evaluation, model saving and other functions, which can achieve the following functions:

1. Specify basic parameters such as training model and epoches
2. Whether to choose **pre-training model**
3. Pick up where you left off
4. **Save the best model** and the last trained model
5. Evaluation of the model: Loss and Accuracy
6. Visualize with **TensorBoard**

#### 1.Start training

In the 'main.py' program, set parameters and models (you can see what models are available in 'models.py') :

Click Run in vscode or type in the command line:

```bash
python3 main.py
```

You can start training, and the effect after starting training is as follows:

If the program is interrupted, set the resume parameter to True, you can continue to train the last model, you can very convenient **as many times as you want to train**.

You can also adjust learning rate, learning rate decay, weight decay, optimizer, and whether to select a pre-trained model

#### 2.tensorboard visualization

Open tensorboard in vscode, or go to the current project folder on the command line and type

```python
tensorboard --logdir runs
```

### Model summary

|          model        |     val acc     |
| :--------------------:| :--------------: |
|         LeNet         |       83%       |
|   LeNet(use dropout)  |       81%       |
|        Alexnet        |       90%       |
|       squeezeNet      |       96%       |
|         resNet        |       95%       |
|    resNet(pretrain)   |       98%       |

## Predict

After the model is trained, you can open 'predict.py' to predict the new picture, given the model to predict and the forecast picture folder:

```python
 model = ResNet34() # Model structure
    modelpath = "./runs/ResNet34_ep=100_bs=32_lr=5e-05_ld=0.9_wd=0/ResNet34_best.pth" # Trained model path
    checkpoint = torch.load(modelpath)  
    model.load_state_dict(checkpoint)  # Loading model parameters
  
    root = "test_pics"
```

Running 'predict.py' stores the csv in the 'output' folder.
