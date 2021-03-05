import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import math
import csv
from yolo import YOLO
import warnings
from deep_sort import build_tracker
# from utils.draw import draw_boxes
from utils.parser import get_config
from utils.draw import draw_boxes,pil_draw_box,pil_draw_box_2
from trafficLine import *
from utils.utils import *
from licence import Licence
import shutil
from detectCar.test import detectCar 
from pyecharts import Line,Pie
from PyQt5.QtGui import *
import json
# from utils.log import get_logger
# from utils.io import write_results


class VideoTracker(object):
    def __init__(self, cfg):
        warnings.filterwarnings("ignore")
        self.cfg = cfg
        #self.args = args
        self.yolo = 0
        self.video_path = 0
        self.outputDir = 0
        self.carLocation1={}
        self.carLocation2={}
        self.carSpeed={}
        self.carDirection={}
        self.carPre={}
        self.virtureLine=[[0,0],[0,0]]
        self.carInCnt=0
        self.carOutCnt=0
        self.inCar=set()
        self.outCar=set()
        self.trafficLine=[]
        self.idx_frame=0
        self.carSpeedLimit=0
        self.filename=0
        self.trafficJamLimit=0
        self.rett=False
        self.carLicense={}
        self.speed=[0,0,0,0,0,0,0,0]
        self.licence=Licence()
        self.detectCar=detectCar()
        self.frameAll=0
        self.vout=cv2.VideoWriter()
        self.carLabel={}
        self.saveVideoFlag=True
        self.displayFlag=True
        self.videoFps=0
        self.videoHeight=0
        self.trafficLine1=[]
        self.trafficLine2=[]
        self.carInfor={}
        self.videoWidth=0
        self.carSpeedLimit=0
        self.trafficJamLimit=0
        self.yolov5weight=0
        self.carFromLeft={}
        self.carFromRight={}
        self.carFromForward={}
        self.carTurn={}
        self.endFlag=False
        self.illegal={}
        self.trafficLightFlag=False
        self.dir=['UP','RIGHT','DOWN',"LEFT"]
        #self.idx_frame
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        
        self.detector = self.yolo
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        #self.class_names = self.detector.class_names
    
    def detectTrafficLine(self,img):
        Ni, Nj = (90, 1300)

        M = np.array([[-1.86073726e-01, -5.02678929e-01,  4.72322899e+02],
                        [-1.39150388e-02, -1.50260445e+00,  1.00507430e+03],
                        [-1.77785988e-05, -1.65517173e-03,  1.00000000e+00]])

        iM = inv(M)
        xy = np.zeros((640,640,2),dtype=np.float32)
        for py in range(640):
            for px in range(640):
                xy[py,px] = np.array([px,py],dtype=np.float32)
        ixy=cv2.perspectiveTransform(xy,iM)
        mpx,mpy = cv2.split(ixy)
        mapx,mapy=cv2.convertMaps(mpx,mpy,cv2.CV_16SC2)
        gray = preprocessing(img)
        canny = cv2.Canny(gray,30,90,apertureSize = 3)
        Amplitude, theta = getGD(canny)
        indices, patches = zip(*sliding_window(Amplitude, theta, patch_size=(Ni, Nj))) #use sliding_window get indices and patches
        labels = predict(patches, False) #predict zebra crossing for every patches 1 is zc 0 is background
        indices = np.array(indices)
        ret, location = getlocation(indices, labels, Ni, Nj)
        return ret,location
    def calculateSpeed(self,location1, location2,cnt,flag=0):
        
        x11,y11,x12,y12=location1
        x21,y21,x22,y22=location2

        w1=x12-x11
        h1=y12-y11
        w2=x22-x21
        h2=y22-y21

        cx1,cy1=x11+w1/2,y11+h1/2
        cx2,cy2=x21+w2/2,y21+h2/2
        dis=math.sqrt(pow(abs(cx2-cx1),2)+pow(abs(cy2-cy1),2))
        h=(h1+h2)/2
        w=(w1+w2)/2
        if w1/h1>=2 or w2/h2>=2 or flag==1:
            dpix=1.8/h
            dis=dis*dpix
            v=dis*3.6/cnt*self.videoFps
            return v

        dpix=7.6/w
        dis=dis*dpix
        v=dis*3.6/cnt*self.videoFps
        return v
    
    def calculateDirection(self,location1, location2):
        x11,y11,x12,y12=location1
        x21,y21,x22,y22=location2

        w1=x12-x11
        h1=y12-y11
        w2=x22-x21
        h2=y22-y21

        cx1,cy1=x11+w1/2,y11+h1/2
        cx2,cy2=x21+w2/2,y21+h2/2
        dx=cx2-cx1
        dy=cy2-cy1
        if dy>0 and 3*abs(dy)>=abs(dx):
            return 2
        if dy<0 and 1.5*abs(dy)>=abs(dx):
            return 0
        if dx>0 and abs(dx)>=abs(dy):
            return 1
        if dx<0 and abs(dx)>=abs(dy):
            return 3
        
        return 0


    def saveVideo(self):
        
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps=self.vdo.get(cv2.CAP_PROP_FPS)

        size= (int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.vout=cv2.VideoWriter(self.outputDir+self.filename+"_processed.mp4",fourcc,fps,size)

    def run(self,path,sinImg,sin2,sinInfo,sinIllegal,config):
        self.carSpeedLimit=config.CARSPEEDLIMIT
        self.trafficJamLimit=config.TRAFFICJAMLIMIT
        self.yolov5weight=config.YOLOWEIGHT
        self.detector = YOLO(self.yolov5weight)


        
        self.video_path=path
        filepath,filename=os.path.split(path)
        self.outputDir=os.getcwd()+"/output/"+filename.split('.')[0]+"/"
        self.filename=filename.split('.')[0]
        

        if os.path.exists(self.outputDir):
            shutil.rmtree(self.outputDir)
        os.mkdir(self.outputDir)

        self.vdo = cv2.VideoCapture(self.video_path)
        self.videoWidth,self.videoHeight=int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.videoFps=self.vdo.get(cv2.CAP_PROP_FPS)
        self.frameAll=self.vdo.get(7)
        results = []
        self.idx_frame = 0
        if self.saveVideoFlag==True:
            self.saveVideo()

        self.trafficLine1=config.trafficline1
        self.trafficLine2=config.trafficline2
        while self.vdo.grab():
            if self.endFlag:
                self.vdo.release()
                self.vout.release()
                self.generate_chart()
                break

            self.idx_frame += 1

                
            #if idx_frame % self.args.frame_interval:
            #    continue
            
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            image=ori_im

            if len(self.trafficLine1)>0 or len(self.trafficLine2)>0:
                self.trafficLine=self.trafficLine1
                pass
            else:
                if self.idx_frame<5 and self.rett == False:
                    #self.rett=True
                    ret,location=self.detectTrafficLine(image)
                    if(ret!=0):
                        self.rett=True
                        self.trafficLine=[location[0][0],location[0][1],location[1][0],location[1][1]]



            self.virtureLine=[0,int(image.shape[0]/2),int(image.shape[1]),int(image.shape[0]/2)]
            cv2.line(image,(0,int(image.shape[0]/2)),(int(image.shape[1]),int(image.shape[0]/2)),color=(10,223,224),thickness=5)
            im1=ori_im.copy()
            im = im1

            
            bbox_xywh, cls_conf, cls_ids,ori_im = self.detector.detect_image(ori_im,self.trafficLine,self.outputDir,self.idx_frame,self.illegal,config)
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]


                for i,bbox in enumerate(bbox_xyxy):
                    x1,y1,x2,y2=bbox_xyxy[i]



                    id=int(identities[i]) if identities is not None else 0  
                    if(self.carLicense.get(id,0)==0):
                        t=self.licence.detectLicence(ori_im[y1:y2,x1:x2],x1,y1)
                        if t!=None:
                            xyxy,label=t
                            self.carLicense[id]=label
                    if self.carLabel.get(id,0)==0 and config.CARCLASSIFY:
                        t=self.detectCar.detect(ori_im[y1:y2,x1:x2])
                        if t!=None:
                            label=t
                            self.carLabel[id]=label

                    if(self.carPre.get(id,0)==0):
                        self.carPre[id]=self.idx_frame
                        self.carLocation1[id]=[x1,y1,x2,y2]
                    elif (self.idx_frame-self.carPre[id])>=12:
                        
                        self.carLocation2[id]=[x1,y1,x2,y2]
                        if self.carLocation1[id][3]<=self.virtureLine[1] and y2>=self.virtureLine[1]:
                            self.inCar.add(id)
                        elif self.carLocation1[id][1]>=self.virtureLine[1] and y1<=self.virtureLine[1]:
                            self.outCar.add(id)
                      
                        pre=self.carDirection.get(id,0)
                        self.carSpeed[id]=self.calculateSpeed(self.carLocation1[id],self.carLocation2[id],self.idx_frame-self.carPre[id])
                        if self.carSpeed[id]>=10:
                            self.carDirection[id]=self.dir[self.calculateDirection(self.carLocation1[id],self.carLocation2[id])]
                        if self.carDirection.get(id,0)=='LEFT' or self.carDirection.get(id,0)=='RIGHT':
                            self.carSpeed[id]=self.calculateSpeed(self.carLocation1[id],self.carLocation2[id],self.idx_frame-self.carPre[id],1)
                        if self.carSpeed.get(id,0)>self.carSpeedLimit:
                            if os.path.exists(self.outputDir+"illegal/"):
                                pass
                            else:
                                os.mkdir(self.outputDir+"illegal")
                            if os.path.exists(self.outputDir+"illegal/overspeed"):
                                pass
                            else:
                                os.mkdir(self.outputDir+"illegal/overspeed")
                        
                            imgTmp=ori_im[y1:y2,x1:x2]
                            imgg=imgTmp.copy()
                            imgg=Image.fromarray(cv2.cvtColor(imgg,cv2.COLOR_BGR2RGB))
                            if isinstance(self.illegal.get(self.idx_frame,0),int):
                                self.illegal[self.idx_frame]={}
                            self.illegal[self.idx_frame].update({'overspeed':str(id)+" "+self.carLicense.get(id,"")+" "+self.carLabel.get(id,"")})
                            imgg.save(self.outputDir+"illegal/overspeed/"+str(self.idx_frame)+"_"+self.carLicense.get(id,'')+".jpg")
                            font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(0.012 * np.shape(ori_im)[1] ).astype('int32'))
                            ori_im=pil_draw_box_2(ori_im,[x1,y1,x2,y2],label="超速",font=font)
                        
                        if pre=='UP':
                            if self.carDirection.get(id,0)=='LEFT':
                                self.carTurn[id]='LEFT'
                            elif self.carDirection.get(id,0)=='RIGHT':
                                self.carTurn[id]='RIGHT'
                        if self.carTurn.get(id,0)=='LEFT' and (self.carFromRight.get(id,0)==True or self.carFromForward.get(id,0)==True) and self.carFromLeft.get(id,0)!=True: 
                            
                            ori_im=pil_draw_box_2(ori_im,[x1,y1,x2,y2],label="未按导向行驶",font=font)
                            if isinstance(self.illegal.get(self.idx_frame,0),int):
                                self.illegal[self.idx_frame]={}
                            self.illegal[self.idx_frame].update({'turnwrong':str(id)+" "+self.carLicense.get(id,"")+" "+self.carLabel.get(id,"")})
                            if os.path.exists(self.outputDir+"illegal/"):
                                pass
                            else:
                                os.mkdir(self.outputDir+"illegal")
                            if os.path.exists(self.outputDir+"illegal/turnwrong"):
                                pass
                            else:
                                os.mkdir(self.outputDir+"illegal/turnwrong")
                            imgTmp=ori_im[y1:y2,x1:x2]
                            imgg=Image.fromarray(cv2.cvtColor(imgTmp,cv2.COLOR_BGR2RGB))
                            imgg.save(self.outputDir+"illegal/turnwrong/"+str(self.idx_frame)+"_"+self.carLicense.get(id,'')+".jpg")
                            
                        elif self.carTurn.get(id,0)=='RIGHT' and (self.carFromLeft.get(id,0)==True or self.carFromForward.get(id,0)==True) and self.carFromRight.get(id,0)!=True:

                            ori_im=pil_draw_box_2(ori_im,[x1,y1,x2,y2],label="未按导向行驶",font=font)
                            if isinstance(self.illegal.get(self.idx_frame,0),int):
                                self.illegal[self.idx_frame]={}
                            self.illegal[self.idx_frame].update({'turnwrong':str(id)+" "+self.carLicense.get(id,"")+" "+self.carLabel.get(id,"")})
                            if os.path.exists(self.outputDir+"illegal/"):
                                pass
                            else:
                                os.mkdir(self.outputDir+"illegal")
                            if os.path.exists(self.outputDir+"illegal/turnwrong"):
                                pass
                            else:
                                os.mkdir(self.outputDir+"illegal/turnwrong")
                            imgTmp=ori_im[y1:y2,x1:x2]
                            imgg=Image.fromarray(cv2.cvtColor(imgTmp,cv2.COLOR_BGR2RGB))
                            imgg.save(self.outputDir+"illegal/turnwrong/"+str(self.idx_frame)+"_"+self.carLicense.get(id,'')+".jpg")

 

                        self.carLocation1[id][0]=self.carLocation2[id][0]
                        self.carLocation1[id][1]=self.carLocation2[id][1]
                        self.carLocation1[id][2]=self.carLocation2[id][2]
                        self.carLocation1[id][3]=self.carLocation2[id][3]
                        self.carPre[id]=self.idx_frame
                    #print(x1,x2,y1,y2)
                    if self.carDirection.get(id,0)=='UP':
                        if len(config.laneline)>0:
                            t=bbox_xyxy[i]
                            x1,y1,x2,y2=t
                            #print(t)
                            tmp=t[:]
                            if self.judge_line_illegal(id,tmp,config):
                                #print(t)
                                print(tmp)
                                x,y,xx,yy=tmp
                                ori_im=pil_draw_box_2(ori_im,tmp,label="车辆压线",font=font)
                                if isinstance(self.illegal.get(self.idx_frame,0),int):
                                    self.illegal[self.idx_frame]={}
                                self.illegal[self.idx_frame].update({'touchline':str(id)+" "+self.carLicense.get(id,"")+" "+self.carLabel.get(id,"")})
                                if os.path.exists(self.outputDir+"illegal/"):
                                    pass
                                else:
                                    os.mkdir(self.outputDir+"illegal")
                                if os.path.exists(self.outputDir+"illegal/touchline"):
                                    pass
                                else:
                                    os.mkdir(self.outputDir+"illegal/touchline")
                                print(x,y,xx,yy)
                                imgTmp=ori_im[y:yy,x:xx]
                                imgg=Image.fromarray(cv2.cvtColor(imgTmp,cv2.COLOR_BGR2RGB))
                                imgg.save(self.outputDir+"illegal/touchline/"+str(self.idx_frame)+"_"+self.carLicense.get(id,'')+".jpg")
                                

                    self.carrunred(id,x1,x2,y1,y2,ori_im) 
                    w=x2-x1
                    h=y2-y1
                    cx=x1+w/2
                    cy=y1+w/2
                    if len(config.leftline):
                        if self.inArea(cx,cy,config.leftline):
                            self.carFromLeft[id]=True
                    if len(config.rightline):
                        if self.inArea(cx,cy,config.rightline):
                            self.carFromRight[id]=True
                    if len(config.forwardline):
                        if self.inArea(cx,cy,config.forwardline):
                            self.carFromForward[id]=True                            

                    self.classify_speed(id)
           

                    self.carInfor[id]=self.carDirection.get(id,'')+","+self.carLicense.get(id,'')+","+str(int(self.carSpeed.get(id,0)))+","+self.carLabel.get(id,'')
                    
                    
                #ori_im = draw_boxes(ori_im, bbox_xyxy, identities,self.carSpeed)
                font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(0.012 * np.shape(ori_im)[1] ).astype('int32'))
                ori_im=pil_draw_box(ori_im,bbox_xyxy,identities,self.carSpeed,self.carLicense,self.carLabel,self.carDirection,font)

                if self.detector.carCnt+self.detector.truckCnt>=self.trafficJamLimit:
                    if isinstance(self.illegal.get(self.idx_frame,0),int):
                        self.illegal[self.idx_frame]={}
                    self.illegal[self.idx_frame].update({'trafficjam':True})
                if self.illegal.get(self.idx_frame,0)!=0:
                    sinIllegal.emit(self.illegal)

            end = time.time()
            ori_im = cv2.putText(ori_im, "carIn: %d carOut:%d"%(len(self.inCar),len(self.outCar)), (0,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            if len(self.trafficLine1)>0 or len(self.trafficLine2)>0:
                if len(self.trafficLine1)>0:
                    plot_one_box(self.trafficLine1, ori_im, label="TrafficLine1", color=(205,210,104), line_thickness=3)
                if len(self.trafficLine2)>0:
                    plot_one_box(self.trafficLine2, ori_im, label="TrafficLine2", color=(205,210,104), line_thickness=3)
            
            elif(self.rett):
                #ori_im=draw_boxes(ori_im,self.trafficLine,"")
                label="TrafficLine"
                self.trafficLine[0]=min(self.videoWidth/6,self.trafficLine[0])
                self.trafficLine[2]=max(self.videoWidth/6*5,self.trafficLine[2])
                plot_one_box(self.trafficLine, ori_im, label=label, color=(205,210,104), line_thickness=3)
            
            if len(config.leftline)>0:
                self.plot_lane(config.leftline,ori_im,"leftline")
                #plot_one_box(config.leftline, ori_im, label="LeftLine", color=(105,200,74), line_thickness=3)
            if len(config.rightline)>0:
                self.plot_lane(config.rightline,ori_im,"rightline")
                #plot_one_box(config.rightline, ori_im, label="RightLine", color=(105,200,74), line_thickness=3)
            if len(config.forwardline)>0:
                self.plot_lane(config.forwardline,ori_im,"forwardline")

            if config.SAVEVIDEO:
                self.vout.write(ori_im)
            self.save_csv_data()
            if self.idx_frame==self.frameAll:
                self.vdo.release()
                self.vout.release()
                self.generate_chart()
            self.emitImg(ori_im,sinImg)
            sin2.emit(int((self.idx_frame/self.frameAll)*100))
            sinInfo.emit(self.carInfor)
            print(self.idx_frame,self.frameAll)


    def plot_lane(self,area,im,label=""):
        
        x1,y1,x2,y2,x3,y3,x4,y4=area
        cv2.putText(im, label, (x1, y1-2), 0, 3, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(im,(x1,y1),(x2,y2),(255,0,0),5)
        cv2.line(im,(x2,y2),(x4,y4),(255,0,0),5)
        cv2.line(im,(x4,y4),(x3,y3),(255,0,0),5)
        cv2.line(im,(x3,y3),(x1,y1),(255,0,0),5)
    
    def classify_speed(self,id):
        carID=id
        if(self.carSpeed.get(carID,0))>=5 and (self.carSpeed.get(carID,0))<10:
            self.speed[0]+=1
        elif(self.carSpeed.get(carID,0))>=10 and (self.carSpeed.get(carID,0))<25:
            self.speed[1]+=1
        elif(self.carSpeed.get(carID,0))>=15 and (self.carSpeed.get(carID,0))<20:
            self.speed[2]+=1
        elif(self.carSpeed.get(carID,0))>=20 and (self.carSpeed.get(carID,0))<25:
            self.speed[3]+=1
        elif(self.carSpeed.get(carID,0))>=25 and (self.carSpeed.get(carID,0))<30:
            self.speed[4]+=1
        elif(self.carSpeed.get(carID,0))>=30 and (self.carSpeed.get(carID,0))<40:
            self.speed[5]+=1
        elif(self.carSpeed.get(carID,0))>=40 and (self.carSpeed.get(carID,0))<50:
            self.speed[6]+=1
        elif(self.carSpeed.get(carID,0))>=50 :
            self.speed[7]+=1      

    def carrunred(self,id,x1,x2,y1,y2,ori_im):
        
        if (self.detector.trafficLightColor=='red' or self.detector.forwardlightColor=='red') and self.carDirection.get(id,0)=='UP' and self.carSpeed.get(id,0)>=10:
            if self.carFromForward.get(id,False)==False:
                return
            if y1<self.trafficLine[3]-10:
                pass
            if os.path.exists(self.outputDir+"illegal/"):
                pass
            else:
                os.mkdir(self.outputDir+"illegal")
            if os.path.exists(self.outputDir+"illegal/carrunred"):
                pass
            else:
                os.mkdir(self.outputDir+"illegal/carrunred")

            imgTmp=ori_im[y1:y2,x1:x2]
            self.illegal[self.idx_frame]={}
            self.illegal[self.idx_frame].update({'carrunred':str(id)+" "+self.carLicense.get(id,"")+" "+self.carLabel.get(id,"")})
            imgg=imgTmp.copy()
            imgg=Image.fromarray(cv2.cvtColor(imgg,cv2.COLOR_BGR2RGB))
            imgg.save(self.outputDir+"illegal/carrunred/"+str(self.idx_frame)+"_"+self.carLicense.get(id,'')+".jpg")
            font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(0.012 * np.shape(ori_im)[1] ).astype('int32'))
            ori_im=pil_draw_box_2(ori_im,[x1,y1,x2,y2],label="    闯红灯",font=font)          

    def save_csv_data(self):
        if os.path.exists(self.outputDir+"csv/"):
            pass
        else:
            os.mkdir(self.outputDir+"csv/")
        if self.idx_frame%24==0:
            with open(self.outputDir+"csv/count.csv",'a',newline='') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow([int(self.idx_frame/24),int(self.detector.carCnt),int(self.detector.personCnt),int(self.detector.motoCnt)])
            with open(self.outputDir+"csv/flow.csv",'a',newline='') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow([int(self.idx_frame/24),int(len(self.inCar)),int(len(self.outCar))])
            with open(self.outputDir+"csv/speed.csv",'a',newline='') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(self.speed)

    def judge_line_illegal(self,id,car,config):
        arr=[]
        for i in range(len(config.laneline)):
            arr.append(config.laneline[i])
            if (i+1)%4==0:
                x1,y1,x2,y2=arr
                arr=[]
                q1=[x1,y1]
                q2=[x2,y2]
                tx1,ty1,tx2,ty2=car
                p1=[tx1,ty2]
                p2=[tx2,ty2]

                if self.isIntersect(p1,p2,q1,q2):
                    return True
        return False

    def det(self,p1,p2):
        x1,y1=p1
        x2,y2=p2
        return x1*y2-y1*x2

    def dcmp(self,a):
        return abs(a)<1e-6

    def dot(self,p1,p2):
        x1,y1=p1
        x2,y2=p2
        return x1*x2+y1*y2

    def on_seg(self,p1,p2,q):
        d1=[p1[0]-q[0],p1[1]-q[1]]
        d2=[p2[0]-q[0],p2[1]-q[1]]
        return self.dcmp(self.det(d1,d2)) and self.dot(d1,d2)<=1e-6


    def intersection(self,p1,p2,q1,q2):
        p2p1=[p2[0]-p1[0],p2[1]-p1[1]]
        q2q1=[q2[0]-q1[0],q2[1]-q1[1]]
        q1p1=[q1[0]-p1[0],q1[1]-p1[1]]
        t=self.det(q2q1,q1p1)/self.det(q2q1,p2p1)
        t2=[p2p1[0]*t,p2p1[1]*t]
        res=[p1[0]+t2[0],p1[1]+t2[1]]
        return res
    
    def isIntersect(self,p1,p2,q1,q2):
        t=self.intersection(p1,p2,q1,q2)
        return self.on_seg(p1,p2,t) and self.on_seg(q1,q2,t)

    def generate_chart(self):
        if os.path.exists(self.outputDir+"chart/"):
            pass
        else:
            os.mkdir(self.outputDir+"chart/")
        
        line=Line("检测目标分析图")
        t1=['time','car','person','moto']

        if os.path.exists(self.outputDir+"csv/count.csv"):
            with open(self.outputDir+"csv/count.csv",'r') as f:
                reader=csv.reader(f)
                result=np.array(list(reader))
                for i in range(len(result[0])):
                    if i==0:
                        continue
                    line.add(t1[i],result[0:,0],result[0:,i])
                line.render(self.outputDir+'chart/count.html')
        t2=['time','CarIn','CarOut']
        line2=Line("流量分析图")
        if os.path.exists(self.outputDir+"csv/flow.csv"):
            with open(self.outputDir+"csv/flow.csv",'r') as f:
                reader=csv.reader(f)
                result=np.array(list(reader))
                for i in range(len(result[0])):
                    if i==0:
                        continue
                    line2.add(t2[i],result[0:,0],result[0:,i])
                line2.render(self.outputDir+"chart/flow.html")
        pie=Pie("速度区间分析图")
        if os.path.exists(self.outputDir+"csv/speed.csv"):
            with open(self.outputDir+"csv/speed.csv",'r') as f:
                reader=csv.reader(f)
                result=np.array(list(reader))
                pie.add("速度区间",["[5km/h,10km/h)","[10km/h,15km/h)","[15km/h,20km/h)","[20km/h,25km/h)","[25km/h,30km/h)","[30km/h,40 km/h)","[40km/h,50 km/h)","[50km/h,+infinity)"],result[-1])
                pie.render(self.outputDir+"chart/speed.html")
        self.saveIllegal()
            #break
    
    def saveIllegal(self):
        if os.path.exists(self.outputDir+"illegal/"):
            pass
        else:
            os.mkdir(self.outputDir+"illegal")
        with open(self.outputDir+"illegal/illegal.json",'w') as f:
            json.dump(self.illegal,f)
    def inArea(self,x,y,area):
        x1,y1,x2,y2,x3,y3,x4,y4=area
        if x>(x1+x2)/2 and x<(x3+x4)/2 and y>y1 and y<y2:
            return True
        return False  



    def emitImg(self,cvImg,sinImg):
        #cvImg=cv2.resize(cvImg,(1024,768))
        height, width, channel = cvImg.shape
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        sinImg.emit(QPixmap(qImg))

    def endDetect(self):
        self.endFlag=True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()
# if __name__ == "__main__":
#     args = parse_args()
#     cfg = get_config()
#     #cfg.merge_from_file(args.config_detection)
#     cfg.merge_from_file(args.config_deepsort)

#     vdo_trk=VideoTracker(cfg, args, video_path="video-02.mp4") 
#     vdo_trk.run()