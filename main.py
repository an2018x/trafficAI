import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from videoPlayer import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import time
from detect import VideoTracker
from picPlayer import PicPlayer
from utils.parser import get_config
import re
import os
import cv2
import json

class Config:
    def __init__(self):
        #self.OUTPUTDIR="./output"
        self.YOLOWEIGHT="./weights/yolov5x.pt"
        self.CARSPEEDLIMIT=25
        self.TRAFFICJAMLIMIT=16
        self.SAVEVIDEO=True
        self.CARCLASSIFY=True
        self.forwardline=[]
        self.leftline=[]
        self.rightline=[]
        self.leftlight=[]
        self.forwardlight=[]
        self.rightlight=[]
        self.trafficline1=[]
        self.trafficline2=[]
        self.preflag=[]
        self.laneline=[]
    def initConfig(self):
        self.forwardline=[]
        self.leftline=[]
        self.rightline=[]
        self.leftlight=[]
        self.forwardlight=[]
        self.rightlight=[]
        self.trafficline1=[]
        self.trafficline2=[]
        self.laneline=[]
        #self.preflag=[]

class preDrawWidget(QWidget):
    def __init__(self, config,pg):
        super().__init__()
        self.pg=pg
        self.vlayout=QVBoxLayout(self)
        self.forwardlineCheck=QCheckBox("前行车道线",self)
        self.leftlineCheck=QCheckBox("左转车道线",self)
        self.rightlineCheck=QCheckBox("右转车道线",self)
        self.leftLigthCheck=QCheckBox("前向红绿灯",self)
        self.rightLigthCheck=QCheckBox("左行红绿灯",self)
        self.forwardLightCheck=QCheckBox("右行红绿灯",self)
        self.trafficline1Check=QCheckBox("近人行道",self)
        self.trafficline2Check=QCheckBox("远人行道",self)
        self.lanelineCheck=QCheckBox("车道线",self)
        self.vlayout.addWidget(self.forwardlineCheck)
        self.vlayout.addWidget(self.leftlineCheck)
        self.vlayout.addWidget(self.rightlineCheck)
        self.vlayout.addWidget(self.leftLigthCheck)
        self.vlayout.addWidget(self.rightLigthCheck)
        self.vlayout.addWidget(self.forwardLightCheck)
        self.vlayout.addWidget(self.trafficline1Check)
        self.vlayout.addWidget(self.trafficline2Check)
        self.vlayout.addWidget(self.lanelineCheck)
        
        self.btn=QPushButton("确定",self)
        self.vlayout.addWidget(self.btn)
        self.setLayout(self.vlayout)
        self.config=config
        self.btn.clicked.connect(self.f1)
    def f1(self):
        
        self.config.preflag=[]
        self.config.preflag.append(self.forwardlineCheck.isChecked())
        self.config.preflag.append(self.leftlineCheck.isChecked())
        self.config.preflag.append(self.rightlineCheck.isChecked())
        self.config.preflag.append(self.leftLigthCheck.isChecked())
        self.config.preflag.append(self.rightLigthCheck.isChecked())
        self.config.preflag.append(self.forwardLightCheck.isChecked())
        self.config.preflag.append(self.trafficline1Check.isChecked())
        self.config.preflag.append(self.trafficline2Check.isChecked())
        self.config.preflag.append(self.lanelineCheck.isChecked())
        self.pg.process()
        self.close()
        



class progressWidget(QWidget):
    def __init__(self,path,screen,livepage,config):
        super().__init__()
        self.screen=screen
        self.path=path
        self.livepage=livepage
        self.config=config
        self.label=QLabel("正在处理视频，请稍后")
        self.setWindowTitle("处理中")
        self.endBtn=QPushButton("停止处理",self)
        self.progressbar=QProgressBar(self)
        self.vlayout=QVBoxLayout(self)
        self.vlayout.addWidget(self.label)
        self.vlayout.addWidget(self.progressbar)
        
        self.vlayout.addWidget(self.endBtn)
        self.setLayout(self.vlayout)
        self.checkWidget=preDrawWidget(self.config,self)
        self.checkWidget.show()
        #self.process()
        self.thread=MyThread(path,config)
        #self.thread.start()
        self.img=0
        self.endBtn.clicked.connect(self.f1)
        self.thread.sin.connect(self.f2)
        self.thread.sin2.connect(self.progressbar.setValue)
        self.thread.sinInfo.connect(self.f3)
        self.thread.sinIllegal.connect(self.f4)
        self.xyxy=[]
        self.i=0

    def f1(self):
        self.thread.videoTracker.endDetect()
        time.sleep(0.5)
        if self.thread.isRunning():
            #self.thread.quit()
            self.thread.quit()
        #self.close()

    def OnMouseAction(self,event,x,y,flags,param):
        if len(self.xyxy)==4 and self.i<8 and self.i>2:
            return
        if len(self.xyxy)==8 and self.i>=0 and self.i<=2:
            return
        if event==cv2.EVENT_LBUTTONDOWN:
            self.xyxy.append(x)
            self.xyxy.append(y)
            #imgTmp=self.img.copy()
            cv2.circle(self.imgTmp, (x, y), 2, (0,250,0), 2)
            cv2.imshow(str(self.i),self.imgTmp)
    def process(self):
        self.config.initConfig()
        self.vdo=cv2.VideoCapture(self.path)
        self.img=None
        while self.vdo.grab():
            
            _, self.img = self.vdo.retrieve()
            
            #if self.img!=None:
            break
        x,y=self.img.shape[0:2]
        self.img=cv2.resize(self.img,(int(y/2),int(x/2)))
        self.vdo.release()
        for i in range(len(self.config.preflag)):
            self.xyxy=[]
            self.i=i
            if self.config.preflag[i]==1:
                cv2.namedWindow(str(i))
                self.imgTmp=self.img.copy()
                cv2.setMouseCallback(str(i),self.OnMouseAction)
                cv2.imshow(str(self.i),self.img)
                #k=cv2.waitKey(-1)
                
                flag=0
                if cv2.waitKey(-1)==ord('y'):
                    if i==0:
                        for i in self.xyxy:
                            self.config.forwardline.append(i*2)
                    elif i==1:
                        for i in self.xyxy:
                            self.config.leftline.append(i*2)
                    elif i==2:
                        for i in self.xyxy:
                            self.config.rightline.append(i*2)
                    elif i==3:
                        for i in self.xyxy:
                            self.config.forwardlight.append(i*2)
                    elif i==4:
                        for i in self.xyxy:
                            self.config.leftlight.append(i*2)
                    elif i==5:
                        for i in self.xyxy:
                            self.config.rightlight.append(i*2)
                    elif i==6:
                        for i in self.xyxy:
                            self.config.trafficline1.append(i*2)
                    elif i==7:
                        for i in self.xyxy:
                            self.config.trafficline2.append(i*2)
                    elif i==8:
                        for i in self.xyxy:
                            self.config.laneline.append(i*2)
                    #print("ok")
                    #self.xyxy=[]
                    flag=1
                    cv2.destroyAllWindows()
                    #break
        self.thread.start()
        


    def f2(self,pix):
        
        pixsize=QSize(int(self.screen.width()),int(self.screen.height()))
        scaledPixmap = pix.scaled(pixsize, Qt.KeepAspectRatio);
        self.screen.setPixmap(scaledPixmap)

    def f3(self,dict):
        for id,value in dict.items():
            if self.livepage.inforList.get(id,0)==0:
                self.livepage.inforList[id]=value
                self.livepage.listWidget.addItem("carID:"+str(id))
            else:
                self.livepage.inforList[id]=value


    def f4(self,dict):
        for id,value in dict.items():
            #print(id)
            if self.livepage.inforList2.get(id,0)==0:
                
                self.livepage.inforList2[id]=value
                self.livepage.listWidget2.addItem("frame:"+str(id))
            else:
                self.livepage.inforList2[id]=value

class MyThread(QThread):
    #trigger = pyqtSignal(str, str)
    sin=pyqtSignal(QPixmap)
    sin2=pyqtSignal(int)
    sinInfo=pyqtSignal(dict)
    sinIllegal=pyqtSignal(dict)
    def __init__(self,path,config):
        
        super(MyThread, self).__init__()
        self.cfg = get_config()
        self.config=config
        self.cfg.merge_from_file("./configs/deep_sort.yaml")
        self.videoTracker=VideoTracker(self.cfg)
        self.videopath=path
        #self.pbar=progressWidget()
        #self.pbar.show()
        
        #pass
    def run(self):
        self.videoTracker.run(self.videopath,self.sin,self.sin2,self.sinInfo,self.sinIllegal,self.config)
        #self.pbar.close()
        #pass
        

class SettingPage(QWidget):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.vlayout=QVBoxLayout(self)
        self.selectFileBtn=QPushButton("选择权重文件",self)
        self.weightLineEdit=QLineEdit(self)

        # self.outputBtn=QPushButton("选择输出路径",self)
        # self.outputLineEdit=QLineEdit(self)

        self.hlayout1=QHBoxLayout()
        self.hlayout2=QHBoxLayout()
        self.hlayout3=QHBoxLayout()
        self.hlayout4=QHBoxLayout()
        self.hlayout5=QHBoxLayout()
        self.submitBtn=QPushButton("保存设置",self)
        self.carspeedEdit=QLineEdit(self)
        self.trafficJamEdit=QLineEdit(self)
        self.checkBox=QCheckBox("保存视频文件",self)
        self.checkBox2=QCheckBox("车辆分类",self)
        self.submitBtn.setMaximumSize(60,40)
        self.submitBtn.setMinimumSize(60,40)
        self.initUI()
        self.submitBtn.clicked.connect(self.saveConfig)
        self.selectFileBtn.clicked.connect(self.f1)
        #self.outputBtn.clicked.connect(self.f2)
        self.readSetting()
        
        
    def f1(self):
        
        (filename,_)=QFileDialog.getOpenFileName(self,'选择权重文件','','')
        self.weightLineEdit.setText(filename)

    # def f2(self):
    #     filename=QFileDialog.getExistingDirectory(self,'选择文件夹','')
    #     self.outputLineEdit.setText(filename)

    def readSetting(self):
        if os.path.exists("./settings.json"):
            with open("settings.json","r") as f:
                dict=json.load(f)
                #self.config.OUTPUTDIR=dict['outputdir']
                self.config.YOLOWEIGHT=dict['yoloweight']
                self.config.CARSPEEDLIMIT=dict['carspeedlimit']
                self.config.TRAFFICJAMLIMIT=dict['trafficjamlimit']
                self.config.SAVEVIDEO=dict['savevideo']
                self.config.CARCLASSIFY=dict['carclassify']
        self.weightLineEdit.setText(self.config.YOLOWEIGHT)
        #self.outputLineEdit.setText(self.config.OUTPUTDIR)
        self.checkBox.setChecked(self.config.SAVEVIDEO)
        self.checkBox2.setChecked(self.config.CARCLASSIFY)
        self.trafficJamEdit.setText(str(self.config.TRAFFICJAMLIMIT))
        self.carspeedEdit.setText(str(self.config.CARSPEEDLIMIT))

        
    def saveConfig(self):

        #self.config.OUTPUTDIR=self.outputLineEdit.text()
        self.config.YOLOWEIGHT=self.weightLineEdit.text()
        self.config.CARSPEEDLIMIT=int(self.carspeedEdit.text())
        self.config.TRAFFICJAMLIMIT=int(self.trafficJamEdit.text())
        self.config.SAVEVIDEO=self.checkBox.isChecked()
        self.config.CARCLASSIFY=self.checkBox2.isChecked()
        with open('settings.json','w') as f:
            setting_dict={'yoloweight':self.config.YOLOWEIGHT,'carspeedlimit':self.config.CARSPEEDLIMIT,'trafficjamlimit':self.config.TRAFFICJAMLIMIT,'savevideo':self.config.SAVEVIDEO,'carclassify':self.config.CARCLASSIFY}
    
            json.dump(setting_dict,f)
        self.readSetting()

    def initUI(self):
        self.label1=QLabel("权重文件设置：")
        self.hlayout1.addWidget(self.label1,Qt.AlignLeft)
        self.hlayout1.addWidget(self.weightLineEdit,Qt.AlignLeft)
        self.hlayout1.addWidget(self.selectFileBtn,Qt.AlignLeft)

        # self.label2=QLabel("输出文件位置：")
        # self.hlayout2.addWidget(self.label2,Qt.AlignLeft)
        # self.hlayout2.addWidget(self.outputLineEdit,Qt.AlignLeft)
        # self.hlayout2.addWidget(self.outputBtn,Qt.AlignLeft)

        self.label3=QLabel("车辆超速阈值：")
        self.hlayout3.addWidget(self.label3,Qt.AlignLeft)
        self.hlayout3.addWidget(self.carspeedEdit,Qt.AlignLeft)

        self.label4=QLabel("交通堵塞阈值：")
        self.hlayout4.addWidget(self.label4,Qt.AlignLeft)
        self.hlayout4.addWidget(self.trafficJamEdit,Qt.AlignLeft)
        
        self.hlayout5.addWidget(self.checkBox,Qt.AlignLeft)
        self.hlayout5.addWidget(self.checkBox2,Qt.AlignLeft)

        self.vlayout.addLayout(self.hlayout1)
        self.vlayout.addLayout(self.hlayout2)
        self.vlayout.addLayout(self.hlayout3)
        self.vlayout.addLayout(self.hlayout4)
        self.vlayout.addLayout(self.hlayout5)
        self.vlayout.addWidget(self.submitBtn,Qt.AlignHCenter)
        #self.setLayout(self.vlayout)

        
        
class LivePage(QMainWindow):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super().__init__(parent=parent, flags=flags)
        self.screenDockWidget=QDockWidget(self)
        self.carInforDockWidget=QDockWidget(self)
        self.illegalInfoDockWidget=QDockWidget(self)

        self.inforList={}
        self.inforList2={}
        #self.illegalscreen=QLabel(self)
        self.listWidget=QListWidget(self)
        self.listWidget2=QListWidget(self)
        #self.addDockWidget(Qt.LeftDockWidgetArea,self.carInforDockWidget)
        self.addDockWidget(Qt.RightDockWidgetArea,self.illegalInfoDockWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea,self.screenDockWidget)
        self.splitDockWidget(self.illegalInfoDockWidget,self.carInforDockWidget,Qt.Vertical)
        
        self.screen=QLabel(self)
        #self.screen.resize(self.width(),self.height())
        self.screenDockWidget.setWidget(self.screen)
        self.carInforDockWidget.setWidget(self.listWidget)
        self.screen.setStyleSheet("background-color: white")
        self.illegalInfoDockWidget.setWidget(self.listWidget2)
        #self.illegalscreen.setStyleSheet("background-color: white")
        
        self.listWidget.itemClicked.connect(self.f1)

        self.listWidget2.itemClicked.connect(self.f2)
    

    def setPic(self,px):
        self.screen.setPixmap(px)
    def f1(self,item):
        QMessageBox.information(self,"carinfo",self.inforList[(int(item.text().split(':')[-1]))])

    def f2(self,item):
        QMessageBox.information(self,"illegalInfo",str(self.inforList2[(int(item.text().split(':')[-1]))]))








class MyClass(QWidget):
    def __init__(self):
        super().__init__()
        self.config=Config()
        #self.cap=cv2.VideoCapture()
        self.continuePlay=True
        self.setWindowIcon(QIcon('./img/icon.ico'))
        self.videoPlayer=VideoPlayer()
        self.settingWidget=SettingPage(self.config)
        self.initUI()
        self.img=np.ndarray(())
        self.btn_addVideo.clicked.connect(self.on_btn_addVideo_clicked)
        self.filename=""
        self.outputDir=""
        self.settingWidget.readSetting()
        
        
        

    def on_btn_addVideo_clicked(self):
        file_name,ok1=QFileDialog.getOpenFileName(self,'选择视频文件','','Video files (*.mp4 *.avi)')
        if file_name=="":
            return
        (filepath,filename)=os.path.split(file_name)
        self.filename=filename
        self.outputDir=os.getcwd()+"/output/"+self.filename.split('_')[0]+"/"
        #print(self.outputDir)
        if os.path.exists(self.outputDir+'chart/count.html'):
            self.browser.load(QUrl(QFileInfo(self.outputDir+'chart/count.html').absoluteFilePath()))
        if(re.match(r"^.*_processed.mp4",filename)!=None):
            self.videoPlayer.setPlayFile(file_name)
        else:
            #print(self.stackWidget.height())
            self.liveWidget.resize(self.stackWidget.width()-2,self.stackWidget.height()-2)
            self.liveWidget.screenDockWidget.setMinimumWidth(self.liveWidget.width()/3*2)
            
            #self.liveWidget.screen.resize(self.liveWidget.width()-2,self.liveWidget.height()-2)
            self.progress=progressWidget(file_name,self.liveWidget.screen,self.liveWidget,self.config)
            self.progress.show()

    def f1(self):
        self.stackWidget.setCurrentIndex(0)
    
    def f2(self):
        self.stackWidget.setCurrentIndex(1)
    
    def f3(self):
        self.stackWidget.setCurrentIndex(2)
        self.picPlayer.setFileName(self.outputDir)
    def f4(self):
        self.stackWidget.setCurrentIndex(3)
    def f5(self):
        self.stackWidget.setCurrentIndex(4)

    def f6(self):
        self.stackWidget.setCurrentIndex(5)

    def changeView(self,x):
        if x==0:

            if os.path.exists(self.outputDir+'chart/count.html'):
                self.browser.load(QUrl(QFileInfo(self.outputDir+'chart/count.html').absoluteFilePath()))
        elif x==1:
            if os.path.exists(self.outputDir+'chart/flow.html'):
                self.browser.load(QUrl(QFileInfo(self.outputDir+'chart/flow.html').absoluteFilePath()))
        elif x==2:
            if os.path.exists(self.outputDir+'chart/speed.html'):
                self.browser.load(QUrl(QFileInfo(self.outputDir+'chart/speed.html').absoluteFilePath()))
        
    def initUI(self):
        self.picPlayer=PicPlayer()
        self.aboutText=QPlainTextEdit()
        self.aboutText.setFont(QFont("等线",15))
        self.liveWidget=LivePage()
        self.aboutText.setPlainText(
            """
            本软件是基于计算机视觉的交通检测软件。\n
            目前实现的功能有车辆、行人、人行道、摩托、交通灯等检测。\n
            可实现车辆跟踪和车辆型号检测与车速估计。\n
            可实现对闯红灯、不按导向行驶、超速等违规行为检测和抓拍。\n
            保存必要数据，实现数据可视化。\n
            """
        )
        self.stackWidget=QStackedWidget(self)
        self.widget2=QWidget(self)
        self.vLayout2=QVBoxLayout()
        self.cbx=QComboBox(self)
        self.cbx.addItem("检测目标数量统计")
        self.cbx.addItem("出入流量统计")
        self.cbx.addItem("车速区间统计")
        self.cbx.setMaximumWidth(30)
        self.cbx.setMaximumWidth(180)
        self.cbx.currentIndexChanged[int].connect(self.changeView)
        self.browser=QWebEngineView(self)
        
        self.vLayout2.addWidget(self.cbx)
        self.vLayout2.addWidget(self.browser)
        self.widget2.setLayout(self.vLayout2)
        self.title=QLabel("智能交通监测系统")
        self.title.setStyleSheet('''QLabel{color:white;font-size:22px;font-family:等线;}''')
        self.setWindowOpacity(0.95)
        pe=QPalette()
        self.setAutoFillBackground(True)
        pe.setColor(QPalette.Window,Qt.lightGray)
        self.setPalette(pe)
        self.setWindowTitle("智能交通监测系统")
        self.hwidget=QWidget(self)
        self.vwidget=QWidget(self)
        self.gwidget=QWidget(self)        
        #dk=app.desktop()
        self.vlayout=QVBoxLayout(self)
        self.setGeometry(300,400,1200,818)
        self.hlayout=QHBoxLayout(self)
        self.grid=QGridLayout()
        watch_icon=QIcon("./img/watch.png")
        self.btn_1=QPushButton(self)
        self.btn_1.setMaximumSize(60,60)
        self.btn_1.setMinimumSize(60,60)
        self.btn_1.setIcon(watch_icon)
        self.btn_1.setIconSize(QSize(58,58))
        self.btn_2=QPushButton(self)
        chart_icon=QIcon("./img/chart.png")
        self.btn_2.setMaximumSize(60,60)
        self.btn_2.setMinimumSize(60,60)
        self.btn_2.setIcon(chart_icon)
        self.btn_2.setIconSize(QSize(58,58))
        self.btn_1.setStyleSheet('''QPushButton{border:none;color:white;font-size:18px;font-family:等线;}
        QPushButton:hover{color:white;
                    border:2px solid #F3F3F5;
                    border-radius:35px;
                    background:darkGray;}''')
        self.btn_2.setStyleSheet('''QPushButton{border:none;color:white;font-size:18px;font-family:等线;}
        QPushButton:hover{color:white;
                    border:2px solid #F3F3F5;
                    border-radius:35px;
                    background:darkGray;}''')
        file_icon=QIcon("./img/file.png")
        self.btn_addVideo=QPushButton(self)
        self.btn_addVideo.setMaximumSize(60,60)
        self.btn_addVideo.setMaximumSize(60,60)
        self.btn_addVideo.setIcon(file_icon)
        self.btn_addVideo.setIconSize(QSize(58,58))
        self.btn_addVideo.setStyleSheet('''QPushButton{border:none;color:white;font-size:18px;font-family:等线;}
        QPushButton:hover{color:white;
                    border:2px solid #F3F3F5;
                    border-radius:35px;
                    background:darkGray;}''')

        self.btn_3=QPushButton(self)
        pic_icon=QIcon("img/pic.png")
        self.btn_3.setMaximumSize(60,60)
        self.btn_3.setMinimumSize(60,60)
        self.btn_3.setIcon(pic_icon)
        self.btn_3.setIconSize(QSize(58,58))
        self.btn_3.setStyleSheet('''QPushButton{border:none;color:white;font-size:18px;font-family:等线;}
        QPushButton:hover{color:white;
                    border:2px solid #F3F3F5;
                    border-radius:35px;
                    background:darkGray;}''')
        self.btn_4=QPushButton(self)
        about_icon=QIcon("img/about.png")
        self.btn_4.setMaximumSize(60,60)
        self.btn_4.setMinimumSize(60,60)
        self.btn_4.setIcon(about_icon)
        self.btn_4.setIconSize(QSize(58,58))
        self.btn_4.setStyleSheet('''QPushButton{border:none;color:white;font-size:18px;font-family:等线;}
        QPushButton:hover{color:white;
                    border:2px solid #F3F3F5;
                    border-radius:35px;
                    background:darkGray;}''')
        self.btn_5=QPushButton(self)
        setting_icon=QIcon("img/setting.png")
        self.btn_5.setMaximumSize(60,60)
        self.btn_5.setMinimumSize(60,60)
        self.btn_5.setIcon(setting_icon)
        self.btn_5.setIconSize(QSize(58,58))
        self.btn_5.setStyleSheet('''QPushButton{border:none;color:white;font-size:18px;font-family:等线;}
        QPushButton:hover{color:white;
                    border:2px solid #F3F3F5;
                    border-radius:35px;
                    background:darkGray;}''')

        self.btn_6=QPushButton(self)
        live_icon=QIcon("img/live.png")
        self.btn_6.setMaximumSize(60,60)
        self.btn_6.setMinimumSize(60,60)
        self.btn_6.setIcon(live_icon)
        self.btn_6.setIconSize(QSize(58,58))
        self.btn_6.setStyleSheet('''QPushButton{border:none;color:white;font-size:18px;font-family:等线;}
        QPushButton:hover{color:white;
                    border:2px solid #F3F3F5;
                    border-radius:35px;
                    background:darkGray;}''')

        self.btn_1.clicked.connect(self.f1)
        self.btn_2.clicked.connect(self.f2)
        self.btn_3.clicked.connect(self.f3)
        self.btn_4.clicked.connect(self.f4)
        self.btn_5.clicked.connect(self.f5)
        self.btn_6.clicked.connect(self.f6)
        self.grid.addWidget(self.btn_1,1,0)
        self.grid.addWidget(self.btn_2,2,0)
        self.grid.addWidget(self.btn_3,3,0)
        self.grid.addWidget(self.btn_4,4,0)
        self.grid.addWidget(self.btn_5,5,0)
        self.grid.addWidget(self.btn_6,0,0)
        self.grid.addWidget(self.btn_addVideo,6,0)
        self.grid.addWidget(self.stackWidget,0,1,7,5)
        self.screeWidget=QWidget(self)
        self.btn_exit=QPushButton("",self)
        self.btn_max=QPushButton("",self)
        self.btn_min=QPushButton("",self)
        self.btn_exit.setMaximumSize(20,20)
        self.btn_max.setMaximumSize(20,20)
        self.btn_min.setMaximumSize(20,20)
        self.btn_exit.setStyleSheet('''QPushButton{background:#F76677;border-radius:10px;}QPushButton:hover{background:red;}''')
        self.btn_max.setStyleSheet('''QPushButton{background:#F7D674;border-radius:10px;}QPushButton:hover{background:yellow;}''')
        self.btn_min.setStyleSheet('''QPushButton{background:#6DDF6D;border-radius:10px;}QPushButton:hover{background:green;}''')
        self.stackWidget.addWidget(self.videoPlayer)
        self.stackWidget.addWidget(self.widget2)
        self.stackWidget.addWidget(self.picPlayer)
        self.stackWidget.addWidget(self.aboutText)
        self.stackWidget.addWidget(self.settingWidget)
        self.stackWidget.addWidget(self.liveWidget)


        self.hlayout.addWidget(self.btn_exit,1)
        self.hlayout.addWidget(self.btn_max,1)
        self.hlayout.addWidget(self.btn_min,1)
        self.hlayout.addWidget(self.title,12)
        self.hwidget.setLayout(self.hlayout)
        self.gwidget.setLayout(self.grid)
        self.vlayout.addWidget(self.hwidget,1)
        self.vlayout.addWidget(self.gwidget,16)
        self.show()
        
 


if __name__=='__main__':
    app=QApplication(sys.argv)
    mc=MyClass()
    app.exec()
    