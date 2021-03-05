import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, carSpeed={},offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = 'ID: %d, SPEED: %d KM/h'%(id,carSpeed.get(id,0))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def pil_draw_box(im,bbox,identities=None,carSpeed={},carLicence={},carLabel={},carDirection={},font=""):
    # 1.读取图片
    #im = Image.open(img_path)

    # 2.获取边框坐标
    # 边框格式　bbox = [xl, yl, xr, yr]
    label1=""
    img=Image.fromarray(im)
    draw = ImageDraw.Draw(img)
    for i,bbox1 in enumerate(bbox):
        #bbox1=bbox[i]
        id = int(identities[i]) if identities is not None else 0  
        label1 = 'ID: %d, SPEED: %d KM/h '%(id,carSpeed.get(id,0))
        
        label1+=(carLicence.get(id,'')+' '+carDirection.get(id,''))
        label_size1 = draw.textsize(label1, font)
        text_origin1 = np.array([bbox1[0], bbox1[1]])
        labelList=carLabel[id].split()
        carlabel0=labelList[0]
        carlabel1=labelList[1]
        carlabel2=labelList[2]
        carlabel_size0 = draw.textsize(carlabel0, font)
        carlabel_size1 = draw.textsize(carlabel1, font)
        carlabel_size2 = draw.textsize(carlabel2, font)
        cartext_origin0 = np.array([bbox1[2], bbox1[1]+carlabel_size0[1]])
        cartext_origin1 = np.array([bbox1[2], bbox1[1]+2*carlabel_size0[1]])
        cartext_origin2 = np.array([bbox1[2], bbox1[1]+3*carlabel_size1[1]])
        # 绘制矩形框，加入label文本
        draw.rectangle([bbox1[0], bbox1[1], bbox1[2], bbox1[3]],outline='red',width=2)
        draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)], fill='red')
        draw.rectangle([tuple(cartext_origin0), tuple(cartext_origin0 + carlabel_size0)], fill='red')
        draw.rectangle([tuple(cartext_origin1), tuple(cartext_origin1 + carlabel_size1)], fill='red')
        draw.rectangle([tuple(cartext_origin2), tuple(cartext_origin2 + carlabel_size2)], fill='red')


        draw.text(text_origin1, str(label1), fill=(255, 255, 255), font=font)
        draw.text(cartext_origin0, str(carlabel0), fill=(255, 255, 255), font=font)
        draw.text(cartext_origin1, str(carlabel1), fill=(255, 255, 255), font=font)
        draw.text(cartext_origin2, str(carlabel2), fill=(255, 255, 255), font=font)

    del draw
    
    #im.save("PIL_img.jpg")
    imgg=np.asarray(img)
    return imgg

def pil_draw_box_2(im,bbox1,label,font=""):
    # 1.读取图片
    #im = Image.open(img_path)

    # 2.获取边框坐标
    # 边框格式　bbox = [xl, yl, xr, yr]
    img=Image.fromarray(im)
    draw = ImageDraw.Draw(img)
    text_origin1 = np.array([bbox1[0], bbox1[1]])
    carlabel0=label
    carlabel_size0 = draw.textsize(carlabel0, font)
    cartext_origin0 = np.array([bbox1[0], bbox1[1]+carlabel_size0[1]])
    # 绘制矩形框，加入label文本
    draw.rectangle([bbox1[0], bbox1[1], bbox1[2], bbox1[3]],outline='blue',width=2)
    draw.rectangle([tuple(cartext_origin0), tuple(cartext_origin0 + carlabel_size0)], fill='blue')
    draw.text(cartext_origin0, str(carlabel0), fill=(255, 255, 255), font=font)
    del draw
    
    #im.save("PIL_img.jpg")
    imgg=np.asarray(img)
    return imgg



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
