import cv2
from utils.datasets import *
from utils.utils import *
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import time
import csv
import lightTest
from utils.draw import draw_boxes,pil_draw_box_2
from licence import Licence


class YOLO(object):
    _defaults = {
        "model_path": 'yolo4_weights.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "model_image_size" : (416, 416, 3),
        "confidence": 0.5,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, weights):
        self.__dict__.update(self._defaults)
        #self.class_names = self._get_class()
        #self.anchors = self._get_anchors()
        #self.generate()

        self.dir=['UP','RIGHT','DOWN',"LEFT"]
        self.currentCarID=0
        self.virtureLine=[[0,0],[0,0]]
        self.carCnt=0
        self.motoCnt=0
        self.personCnt=0
        self.truckCnt=0
        self.flag=False
        self.trafficLine=None
        self.trafficLight=[0,0,0,0]
        self.curpath=0
        self.trafficLightColor=None
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=torch.load(weights,map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.licence=Licence()
        self.leftlight=[]
        self.forwardlight=[]
        self.rightlight=[]
        self.leftlightColor=0
        self.forwardlightColor=0
        self.rightlightColor=0
        self.carDirection={}
        self.config=None


    def detect_image(self,img,trafficline,path,idx_frame,illegal,config):
        self.config=config
        self.leftlight=self.config.leftlight
        #print(self.leftlight)
        self.forwardlight=self.config.forwardlight
        self.rightlight=self.config.rightlight
        
        self.trafficLine=trafficline
        self.curpath=path
        #(filepath,filename)=os.path.split(path)
        self.personCnt=self.carCnt=self.motoCnt=self.truckCnt=0
        im0=img.copy()
        image=im0
        half=self.device.type!='cpu'
        if half:
            self.model.half()
        img = letterbox(im0,new_shape=(640,640))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)
        
        cars=[]
        return_boxs=[]
        return_class_names=[]
        return_scores=[]
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                #print(det)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                # Write results
                for *xyxy, conf, cls in det:
                    
                    c=cls
                    x=int(xyxy[0])
                    y=int(xyxy[1])
                    w=int(xyxy[2]-xyxy[0])
                    h=int(xyxy[3]-xyxy[1])
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    if (self.config.preflag[3]==1 or self.config.preflag[4]==1 or self.config.preflag[5]==1) :
                        #self.flag=True
                        pass
                    else:
                        if c==9:
                            if self.flag:
                                if y<self.trafficLight[1]:
                                    self.trafficLight=xyxy
                                w=int(xyxy[2]-xyxy[0])
                                h=int(xyxy[3]-xyxy[1])
                                if w>h:
                                    self.config.leftlight=self.leftlight=[xyxy[0],xyxy[1],xyxy[0]+w/2,xyxy[3]]
                                    self.config.forwardlight=self.forwardlight=[xyxy[0]+w/2,xyxy[1],xyxy[2],xyxy[3]]

                                continue
                            self.flag=True
                            self.trafficLight=xyxy
                            print(self.trafficLight)
                            w=int(xyxy[2]-xyxy[0])
                            h=int(xyxy[3]-xyxy[1])
                            if w>h:
                                self.config.leftlight=self.leftlight=[xyxy[0],xyxy[1],xyxy[0]+w/2,xyxy[3]]
                                self.config.forwardlight=self.forwardlight=[xyxy[0]+w/2,xyxy[1],xyxy[2],xyxy[3]]
                                print(self.config.leftlight)
                                print(self.config.forwardlight)
                            continue

                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

                    if c==2:
                        self.carCnt=self.carCnt+1
                    if c==0:
                        self.personCnt=self.personCnt+1
                        if os.path.exists(self.curpath+"illegal/"):
                            pass
                        else:
                            os.mkdir(self.curpath+"illegal")
                        if os.path.exists(self.curpath+"illegal/runred"):
                            pass
                        else:
                            os.mkdir(self.curpath+"illegal/runred")
                        
                        if self.trafficLine!=None:
                            if self.trafficLightColor=='green' and x>=self.trafficLine[0] and x+w<=self.trafficLine[2] and conf>0.6 and (h/w>=1.6):

                                if idx_frame%8==0:
                                    imgTmp=im0[y:y+h,x:x+w]
                                    cv2.imwrite(self.curpath+"illegal/runred/"+str(idx_frame)+".jpg",imgTmp)
                                    if isinstance(illegal.get(idx_frame,0),int):
                                        illegal[idx_frame]={}
                                
                                    illegal[idx_frame].update({'runred':True})
                                font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(0.012 * np.shape(im0)[1] ).astype('int32'))
                                im0=pil_draw_box_2(im0,[x,y,x+w,y+h],label="闯红灯",font=font)
                                
                                
                    if c==3:
                        self.motoCnt=self.motoCnt+1
                    if c==7:
                        self.truckCnt=self.truckCnt+1



                    if c != 2 and c != 7:
                        continue
                    
                    if(y+h<im0.shape[0]-12):
                        #continue
                        return_boxs.append([x+w/2,y+h/2,w,h])
                        return_class_names.append(self.names[int(cls)])
                        return_scores.append(conf)
        
        if len(self.config.leftlight)>0 or len(self.config.forwardlight)>0 or len(self.config.rightlight)>0:
            #print(self.config.leftlight)
            
            if len(self.config.leftlight)>0:
                x1=int(self.config.leftlight[0])
                y1=int(self.config.leftlight[1])
                x2=int(self.config.leftlight[2])
                y2=int(self.config.leftlight[3])
                imgTmp=im0[y1:y2,x1:x2]
                w=x2-x1
                h=y2-y1
                if w>h:
                    imgTmp=self.rotate_bound(imgTmp,90)
                    #cv2.imwrite("test.jpg",imgTmp)
                self.leftlightColor=lightTest.detectImg(imgTmp)
                if self.leftlightColor=='green':
                    plot_one_box(self.leftlight, im0, label=self.leftlightColor, color=(0,255,0), line_thickness=3)
                elif self.leftlightColor=='red':
                    plot_one_box(self.leftlight, im0, label=self.leftlightColor, color=(255,0,0), line_thickness=3)
                elif self.leftlightColor=='yellow':
                    plot_one_box(self.leftlight, im0, label=self.leftlightColor, color=(255,255,0), line_thickness=3)
            if len(self.config.forwardlight)>0:
                x1=int(self.config.forwardlight[0])
                y1=int(self.config.forwardlight[1])
                x2=int(self.config.forwardlight[2])
                y2=int(self.config.forwardlight[3])
                imgTmp=im0[y1:y2,x1:x2]
                w=x2-x1
                h=y2-y1
                #cv2.imwrite("out/"+str(idx_frame)+".jpg",imgTmp)
                if w>h:
                    imgTmp=self.rotate_bound(imgTmp,90)
                
                self.forwardlightColor=lightTest.detectImg(imgTmp)
                if self.forwardlightColor=='green':
                    plot_one_box(self.forwardlight, im0, label=self.forwardlightColor, color=(0,255,0), line_thickness=3)
                elif self.forwardlightColor=='red':
                    plot_one_box(self.forwardlight, im0, label=self.forwardlightColor, color=(255,0,0), line_thickness=3)
                elif self.forwardlightColor=='yellow':
                    plot_one_box(self.forwardlight, im0, label=self.forwardlightColor, color=(255,255,0), line_thickness=3)
            if len(self.config.rightlight)>0:
                x1=int(self.config.rightlight[0])
                y1=int(self.config.rightlight[1])
                x2=int(self.config.rightlight[2])
                y2=int(self.config.rightlight[3])
                imgTmp=im0[y1:y2,x1:x2]
                w=x2-x1
                h=y2-y1
                if w>h:
                    imgTmp=self.rotate_bound(imgTmp,90)
                self.rightlightColor=lightTest.detectImg(imgTmp)
                if self.rightlightColor=='green':
                    plot_one_box(self.rightlight, im0, label=self.rightlightColor, color=(0,255,0), line_thickness=3)
                elif self.rightlightColor=='red':
                    plot_one_box(self.rightlight, im0, label=self.rightlightColor, color=(255,0,0), line_thickness=3)
                elif self.rightlightColor=='yellow':
                    plot_one_box(self.rightlight, im0, label=self.rightlightColor, color=(255,255,0), line_thickness=3)
            
        
        
        elif self.flag==True:
            x1=int(self.trafficLight[0])
            y1=int(self.trafficLight[1])
            x2=int(self.trafficLight[2])
            y2=int(self.trafficLight[3])

            w=x2-x1
            h=y2-y1
            imgLight=im0[y1:y2,x1:x2]
            if w>h:
                imgLight=self.rotate_bound(imgLight,90)
            
            
            self.trafficLightColor=lightTest.detectImg(imgLight)
            if self.trafficLightColor=='green':
                plot_one_box(self.trafficLight, im0, label=self.trafficLightColor, color=(0,255,0), line_thickness=3)
            elif self.trafficLightColor=='red':
                plot_one_box(self.trafficLight, im0, label=self.trafficLightColor, color=(255,0,0), line_thickness=3)
            elif self.trafficLightColor=='yellow':
                plot_one_box(self.trafficLight, im0, label=self.trafficLightColor, color=(255,255,0), line_thickness=3)


        im0 = cv2.putText(im0, "moto: %d  car: %d person:%d truck: %d"%(self.motoCnt,self.carCnt,self.personCnt,self.truckCnt), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        return np.array(return_boxs),np.array(return_scores),np.array(return_class_names),im0

                    #print(xyxy)
                   

    def rotate_bound(self,image,angle):
        #获取图像的尺寸
        #旋转中心
        (h,w) = image.shape[:2]
        (cx,cy) = (w/2,h/2)
        
        #设置旋转矩阵
        M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
        cos = np.abs(M[0,0])
        sin = np.abs(M[0,1])
        
        # 计算图像旋转后的新边界
        nW = int((h*sin)+(w*cos))
        nH = int((h*cos)+(w*sin))
        
        # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
        M[0,2] += (nW/2) - cx
        M[1,2] += (nH/2) - cy
        
        return cv2.warpAffine(image,M,(nW,nH))

    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)




# if __name__=='__main__':
#     img=cv2.imread("./bus.jpg")
#     yolo=YOLO()
#     yolo.detect_image(img,'weights/yolov5s.pt')