#model_ft=


import torch
import torchvision.models as models
import os
import json
import torch.nn as nn
dataset_dir = "../input/car_data/car_data/"
import PIL.Image as Image
import torchvision.transforms as transforms
from IPython.display import display
import cv2

class detectCar(object):

    def __init__(self):
        self.classes=self.readclasses()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=models.resnet34()
        num_ftrs=self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 196)
        self.model = self.model.to(self.device)
        #self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load("./detectCar/carClassify.pth"))
        self.loader = transforms.Compose([transforms.Resize((400, 400)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def find_classes(self,dir=dataset_dir+"train"):
        classes=os.listdir(dir)
        classes.sort()
        class_to_idx={classes[i]:i for i in range(len(classes))}
        return classes,class_to_idx

    def detect(self,img=None):
        img2=img.copy()
        image = Image.fromarray(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
        image = self.loader(image).float()
        image = torch.autograd.Variable(image,requires_grad=True)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        output = self.model(image)
        conf,predicted = torch.max(output.data,1)
        #display(Image.open(dataset_dir+"test/Mercedes-Benz C-Class Sedan 2012/01977.jpg"))
        return self.classes[predicted.item()]

    def writeclasses(self):
        classes,c_to_idx=find_classes()
        with open("classes.json","w") as f:
            json.dump(classes,f)

    def readclasses(self):
        with open("./detectCar/classes.json","r") as f:
            classes=json.load(f)
        return classes

    # if __name__=='__main__':
    #     #writeclasses()
    #     detectCar()
    #     #pass
    #     `