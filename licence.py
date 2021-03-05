import sys
sys.path.append('./LPRNet')
sys.path.append('./MTCNN')
from PyQt5.QtWidgets import QProgressBar
from LPRNet_Test import *
from MTCNN import *
import numpy as np
import torch
import time
import cv2
import torch.nn as nn
import torch
#from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
from utils.utils import plot_one_box
import time
import re
import csv
import shutil
class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(), #13
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=256+class_num+128+64, out_channels=self.class_num, kernel_size=(1,1), stride=(1,1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                #print("intermediate feature map {} shape is: ".format(i), x.shape)
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            #print("after globel context {} shape is: ".format(i), f.shape)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        #print("after container shape is: ", x.shape)
        logits = torch.mean(x, dim=2)

        return logits





class Licence(object):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.yolo = YOLO()
        self.STN = STNet()
        self.STN.to(self.device)
        self.STN.load_state_dict(torch.load('LPRNet/weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
        self.STN.eval()
        self.lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
        self.lprnet.to(self.device)
        self.lprnet.load_state_dict(torch.load('LPRNet/weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
        self.lprnet.eval()
        self.CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
        '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
        '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
        '新',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z', 'I', 'O', '-'
        ]



    def convert_image(self,inp):
        # convert a Tensor to numpy image
        inp = inp.squeeze(0).cpu()
        inp = inp.detach().numpy().transpose((1,2,0))
        inp = 127.5 + inp/0.0078125
        inp = inp.astype('uint8') 

        return inp



    def decode(self,preds, CHARS):
        # greedy decode
        pred_labels = list()
        labels = list()
        for i in range(preds.shape[0]):
            pred = preds[i, :, :]
            pred_label = list()
            for j in range(pred.shape[1]):
                pred_label.append(np.argmax(pred[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = pred_label[0]
            for c in pred_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            pred_labels.append(no_repeat_blank_label)
            
        for i, label in enumerate(pred_labels):
            lb = ""
            for i in label:
                lb += CHARS[i]
            labels.append(lb)
        
        return labels, np.array(pred_labels) 

    def detectLicence(self,img,x,y):
        # 格式转变，BGRtoRGB
        input = img
        bboxes = create_mtcnn_net(input, (50, 15), self.device, p_model_path='MTCNN/weights/pnet_Weights', o_model_path='MTCNN/weights/onet_Weights')
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :4]
            x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]      
            w = int(x2 - x1 + 1.0)
            h = int(y2 - y1 + 1.0)
            img_box = np.zeros((h, w, 3))
            t1=(y1-y2)/10
            t2=(x2-x1)/10
            xyxy=[x1,y1,x2,y2]
            
            img_box = img[y1:y2, x1:x2, :]
            if img_box is None:
                continue



            try:
                im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
            except:
                continue
            im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
            data = torch.from_numpy(im).float().unsqueeze(0).to(self.device)  # torch.Size([1, 3, 24, 94]) 
            transfer = self.STN(data)
            
            preds = self.lprnet(transfer)
            transformed_img = convert_image(transfer)
            preds = preds.cpu().detach().numpy()  # (1, 68, 18)
            labels, pred_labels = self.decode(preds, self.CHARS)            
            # draw.rectangle(
            #             [x1,y1,x2,y2],outline=(0,0,255))
            #if(labels!=None):
            if re.match(r'^[\u4e00-\u9fa5][A-Z0-9]{6}$',labels[0]) != None:
            #在图片上绘制中文
                pass
                #plot_one_box(xyxy, img, label=labels[0], color=Color.YELLOW, line_thickness=3)
            else:
                labels[0]=""
            #xyxy=[x+x1,y+y1,x+x2,y+y2]
            #plot_one_box(xyxy, img, label=labels[0], color=(100,205,255), line_thickness=3)
            
            return xyxy,labels[0]
            

            #return xyxy,labels[0]


