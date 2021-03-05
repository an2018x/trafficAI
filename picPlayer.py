from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sys
import os

class PicPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.filename=''
        self.picList=[]
        self.currentId=0
        self.outputDir=''
        self.cbx.currentIndexChanged[int].connect(self.changeView)
        self.forwardBtn.clicked.connect(self.forward)
        self.backBtn.clicked.connect(self.back)
        self.screen.setStyleSheet("background-color: white;")

    def setFileName(self,filename):
        #self.filename=filename
        self.outputDir=filename
        self.outputDir+='illegal/'
    def changeView(self,x):
        if x==0:
            if os.path.exists(self.outputDir+"runred"):
                print("yes")
                self.picList=os.listdir(self.outputDir+"runred")
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"runred/"+self.picList[i]
                    #print(i)
                self.currentId=0
                if(len(self.picList)==0):
                    return
                else:
                    pix=QPixmap(self.picList[self.currentId])
                    self.screen.setPixmap(pix)
                    t=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    str1=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    
                    self.title.setText(str1)
                #print(self.picList)
            else:
                os.mkdir(self.outputDir+"runred")
                self.picList=os.listdir(self.outputDir+"runred")   
                self.currentId=0 
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"runred/"+self.picList[i]
                if(len(self.picList)==0):
                    return         
        elif x==1:
            if os.path.exists(self.outputDir+"carrunred"):
                self.picList=os.listdir(self.outputDir+"carrunred")
                self.currentId=0
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"carrunred/"+self.picList[i]
                if(len(self.picList)==0):
                    return
                else:
                    pix=QPixmap(self.picList[self.currentId])
                    self.screen.setPixmap(pix)
                    t=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    str1=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    self.title.setText(str1)
            else:
                os.mkdir(self.outputDir+"carrunred")
                self.picList=os.listdir(self.outputDir+"carrunred")   
                self.currentId=0 
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"carrunred/"+self.picList[i]
                if(len(self.picList)==0):
                    return                     
        elif x==2:
            if os.path.exists(self.outputDir+"overspeed"):
                self.picList=os.listdir(self.outputDir+"overspeed")   
                self.currentId=0 
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"overspeed/"+self.picList[i]
                if(len(self.picList)==0):
                    return
                else:
                    pix=QPixmap(self.picList[self.currentId])
                    self.screen.setPixmap(pix)
                    t=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    str1=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    self.title.setText(str1)
            else:
                os.mkdir(self.outputDir+"overspeed")
                self.picList=os.listdir(self.outputDir+"overspeed")   
                self.currentId=0 
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"overspeed/"+self.picList[i]
                if(len(self.picList)==0):
                    return     
        elif x==3:
            if os.path.exists(self.outputDir+"touchline"):
                self.picList=os.listdir(self.outputDir+"touchline")   
                self.currentId=0 
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"touchline/"+self.picList[i]
                if(len(self.picList)==0):
                    return
                else:
                    pix=QPixmap(self.picList[self.currentId])
                    self.screen.setPixmap(pix)
                    t=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    str1=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    self.title.setText(str1)
            else:
                os.mkdir(self.outputDir+"touchline")
                self.picList=os.listdir(self.outputDir+"touchline")   
                self.currentId=0 
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"touchline/"+self.picList[i]
                if(len(self.picList)==0):
                    return     
        elif x==4:
            if os.path.exists(self.outputDir+"turnwrong"):
                self.picList=os.listdir(self.outputDir+"turnwrong")   
                self.currentId=0 
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"turnwrong/"+self.picList[i]
                if(len(self.picList)==0):
                    return
                else:
                    pix=QPixmap(self.picList[self.currentId])
                    self.screen.setPixmap(pix)
                    t=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    str1=self.picList[self.currentId].split("/")[-1].split(".")[0]
                    self.title.setText(str1)         
            else:
                os.mkdir(self.outputDir+"turnwrong")
                self.picList=os.listdir(self.outputDir+"turnwrong")   
                self.currentId=0 
                for i in range(len(self.picList)):
                    self.picList[i]=self.outputDir+"turnwrong/"+self.picList[i]
                if(len(self.picList)==0):
                    return     
        
    def forward(self):
        if(len(self.picList)==0):
            return
        self.currentId+=1
        if self.currentId==len(self.picList):
            self.currentId=0
        pix=QPixmap(self.picList[self.currentId])
        self.screen.setPixmap(pix)
        t=self.picList[self.currentId].split("/")[-1].split(".")[0]
        str1=self.picList[self.currentId].split("/")[-1].split(".")[0]
        self.title.setText(str1)

    def back(self):
        if(len(self.picList)==0):
            return
        self.currentId-=1
        if self.currentId==-1:
            self.currentId=len(self.picList)-1
        pix=QPixmap(self.picList[self.currentId])
        self.screen.setPixmap(pix)
        t=self.picList[self.currentId].split("/")[-1].split(".")[0]
        str1=self.picList[self.currentId].split("/")[-1].split(".")[0]
        self.title.setText(str1)

    def initUI(self):
        self.vLayout=QVBoxLayout()
        self.cbx=QComboBox(self)
        self.cbx.addItem("行人闯红灯")
        self.cbx.addItem("车辆闯红灯")
        self.cbx.addItem("车辆超速")
        self.cbx.addItem("车辆压线")
        self.cbx.addItem("未按导向行驶")
        self.title=QLabel(self)
        self.title.setMaximumHeight(30)
        self.title.setAlignment(Qt.AlignCenter)
        self.screen=QLabel(self)
        self.cbx.setMaximumWidth(30)
        self.cbx.setMaximumWidth(180)
        self.screen.setMinimumSize(600,480)
        self.screen.setAlignment(Qt.AlignCenter)
        self.btnGroup=QWidget(self)
        self.btnGroupLayout=QHBoxLayout(self)
        self.forwardBtn=QPushButton("forward",self)
        self.forwardBtn.setMaximumWidth(120)
        self.backBtn=QPushButton("back",self)
        self.backBtn.setMaximumWidth(120)
        self.btnGroupLayout.addWidget(self.backBtn)
        self.btnGroupLayout.addWidget(self.forwardBtn)
        self.btnGroup.setLayout(self.btnGroupLayout)
        self.vLayout.addWidget(self.cbx)
        self.vLayout.addWidget(self.title)
        self.vLayout.addWidget(self.screen)

        self.vLayout.addWidget(self.btnGroup)
        self.setLayout(self.vLayout)
        



