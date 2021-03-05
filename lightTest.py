import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,models,transforms
import torchvision
import os
import cv2
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 1 input image channel 6 output channels 5*5 square convolution
        #kernal
        # 60*32*3->56*28*6
        self.conv1=nn.Conv2d(3,6,5)
        
        self.conv2=nn.Conv2d(6,16,5)
        # an affine operation y=wx+b
        self.fc1=nn.Linear(12*5*16,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,3)
        
    def forward(self,x):
        # Max pooling over a (2,2) window
        # 56*28*6->28*14*6
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # 24*10*16->12*5*16
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,16*5*12)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        x=F.log_softmax(x,dim=1)
        return x
    
    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features


image_size=(60,32)

process_transform=transforms.Compose([transforms.Resize(image_size),
transforms.ToTensor(),
transforms.Normalize([0.5,0.5,0.5], [0.5, 0.5, 0.5])])
classes=[
    "red","green","yellow"
]

def detectLight(imgPath):
    net=Net()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    net.load_state_dict(torch.load("lightWeight.pth"))
    img=Image.open(imgPath)
    img_tensor=process_transform(img)
    img_tensor.unsqueeze_(0)
    img_tensor=img_tensor.to(device)
    out=net(img_tensor)
    _,predicted=torch.max(out,1)
    #print(classes[predicted[0]])
    
def detectImg(img):
    img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #img=Image.fromarray(img)
    net=Net()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    net.load_state_dict(torch.load("lightWeight.pth"))
    img_tensor=process_transform(img)
    img_tensor.unsqueeze_(0)
    img_tensor=img_tensor.to(device)
    out=net(img_tensor)
    _,predicted=torch.max(out,1)
    #print(classes[predicted[0]])
    return classes[predicted[0]]

def detectImg2(img):
    img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    net=Net()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    net.load_state_dict(torch.load("lightWeight_res.pth"))
    img_tensor=process_transform(img)
    img_tensor.unsqueeze_(0)
    img_tensor=img_tensor.to(device)
    out=net(img_tensor)
    _,predicted=torch.max(out,1)
    #print(classes[predicted[0]])
    return classes[predicted[0]]



# detectLight("test03.jpg")

# img=cv2.imread("test01.jpg")
# detectImg(img)

