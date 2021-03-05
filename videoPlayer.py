from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sys

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selectButton.clicked.connect(self.playVideo)
        self.player.positionChanged.connect(self.changeSlide)
        self.videoSlide.sliderPressed.connect(self.player.pause)
        self.videoSlide.sliderReleased.connect(self.player.play)
        self.videoSlide.sliderMoved.connect(self.changeProgress)
        self.startBtn.clicked.connect(self.player.play)
        self.pauseBtn.clicked.connect(self.player.pause)
        self.stopBtn.clicked.connect(self.player.stop)
        self.videoWidget.setStyleSheet("background-color:white;")
        #self.videoWidget.set
    def changeProgress(self):
        self.player.pause()
        self.player.setPosition((self.videoSlide.value()/100)*self.player.duration())
        self.player.play()
    def playVideo(self):
        self.player.stop()
        self.player.setVideoOutput(self.videoWidget)
        self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取视频文件
        self.player.play()
        
    def changeSlide(self,position):
        self.videoLength=self.player.duration()+0.1
        self.videoSlide.setValue(round((position/self.videoLength)*100))
        
    def setPlayFile(self,file_name):
        print(file_name)
        self.player.stop()
        self.player.setVideoOutput(self.videoWidget)
        self.player.setMedia(QMediaContent(QUrl(file_name)))  # 选取视频文件
        self.player.play()

    def initUI(self):
        self.btnGroup=QWidget(self)
        self.btnGroupLayout=QHBoxLayout(self)
        self.pauseBtn=QPushButton("暂停",self)
        self.startBtn=QPushButton("开始",self)
        self.stopBtn=QPushButton("停止",self)
        self.selectButton=QPushButton("打开文件",self)
        self.btnGroupLayout.addWidget(self.selectButton)
        self.btnGroupLayout.addWidget(self.pauseBtn)
        self.btnGroupLayout.addWidget(self.startBtn)
        self.btnGroupLayout.addWidget(self.stopBtn)
        self.btnGroup.setLayout(self.btnGroupLayout)


        self.vLayout=QVBoxLayout(self)
        self.videoWidget=QVideoWidget(self)
        self.videoSlide=QSlider(Qt.Horizontal,self)
        self.videoSlide.setStyleSheet(
            """
            QSlider{
                border:15px solid darkGray;
            }
            QSlider::add-page:horizontal
            {   
                background-color: rgb(87, 97, 106);
                
                height:4px;
            }
            QSlider::sub-page:horizontal
            {
                background-color: rgb(37, 168, 198);
                height:4px;
            }
            QSlider::groove:horizontal
            {
                background:transparent;
                height:6px;
            }
            QSlider::handle:Horizontal 
            {
                height: 30px;
                width:8px;
                border-image: url(img/handle.png);
                margin: -8 0px; 
            }


            """
        )
        self.player=QMediaPlayer(self)
        
        self.vLayout.addWidget(self.videoWidget,8)
        self.vLayout.addWidget(self.btnGroup,1)
        self.vLayout.addWidget(self.videoSlide,1)
        self.setLayout(self.vLayout)



# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     player=VideoPlayer();
#     player.show()
#     sys.exit(app.exec_())