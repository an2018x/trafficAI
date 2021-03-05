#coding=utf-8
import time
import os
import cv2
import numpy as np
from numpy.linalg import inv
def sliding_window(img1, img2, patch_size=(100,302), istep=50):#, jstep=1, scale=1.0):
    """
get patches and thier upper left corner coordinates
The size of the sliding window is currently fixed.
patch_size: sliding_window's size'
istep: Row stride
    """
    Ni, Nj = (int(s) for s in patch_size)
    for i in range(0, img1.shape[0] - Ni+1, istep):
        #for j in range(0, img1.shape[1] - Nj, jstep):
            #patch = (img1[i:i + Ni, j:j + Nj], img2[i:i + Ni, j:j + Nj])
        patch = (img1[i:i + Ni, 39:341], img2[i:i + Ni, 39:341])
        yield (i, 39), patch

def predict(patches, DEBUG):
    """
predict zebra crossing for every patches 1 is zc 0 is background
    """
    #print(len(patches))
    labels = np.zeros(len(patches))
    index = 0
    for Amplitude, theta in patches:
        mask = (Amplitude>25).astype(np.float32)
        h, b = np.histogram(theta[mask.astype(np.bool)], bins=range(0,80,5))
        low, high = b[h.argmax()], b[h.argmax()+1]
        newmask = ((Amplitude>25) * (theta<=high) * (theta>=low)).astype(np.float32)
        value = ((Amplitude*newmask)>0).sum()

        if value > 1500:
            labels[index] = 1
        index += 1
        if(DEBUG):
            print(h) 
            print(low, high)
            print(value)
            cv2.imshow("newAmplitude", Amplitude*newmask)
            cv2.waitKey(0)
            
    return labels

def preprocessing(img):
    """
Take the blue channel of the original image and filter it smoothly    
    """
    kernel1 = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    gray = img[:,:,0]
    gray = cv2.medianBlur(gray,5)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel1,iterations=4)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2,iterations=3)
    return gray

def getGD(canny):
    """
return gradient mod and direction 
    """
    sobelx=cv2.Sobel(canny,cv2.CV_32F,1,0,ksize=3)
    sobely=cv2.Sobel(canny,cv2.CV_32F,0,1,ksize=3)
    theta = np.arctan(np.abs(sobely/(sobelx+1e-10)))*180/np.pi
    Amplitude = np.sqrt(sobelx**2+sobely**2)
    mask = (Amplitude>30).astype(np.float32)
    Amplitude = Amplitude*mask
    return Amplitude, theta

def getlocation(indices, labels, Ni, Nj):
    """
return if there is a zebra cossing
if true, Combine all the rectangular boxes as its position
assume a picture has only one zebra crossing
    """
    zc = indices[labels == 1]
    if len(zc) == 0:
        return 0, None
    else:
        xmin = int(min(zc[:,1]))
        ymin = int(min(zc[:,0]))
        xmax = int(xmin + Nj)
        ymax = int(max(zc[:,0]) + Ni)
        return 1, ((xmin, ymin), (xmax, ymax))


if __name__ == "__main__":

    DEBUG = False #if False, won't draw all step

    Ni, Nj = (90, 1600)
    
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
    
    cap = cv2.VideoCapture("./video-02.mp4")
    time.sleep(1)
    NUM_FRAMES = int(cap.get(7))
    for ii in range(NUM_FRAMES):
        print("frame: ", ii)
        # Load frame from the camera
        ret, frame = cap.read()
        img=frame
        #img = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        #img = cv2.resize(img, (400,400))
        gray = preprocessing(img)
        # if DEBUG:
        #     cv2.imshow("gray", gray)
    
        canny = cv2.Canny(gray,30,90,apertureSize = 3)
        # if DEBUG:
        #     cv2.imshow("canny",canny)
    
        Amplitude, theta = getGD(canny)
        # if DEBUG:
        #     cv2.imshow("Amplitude", Amplitude)
    
        indices, patches = zip(*sliding_window(Amplitude, theta, patch_size=(Ni, Nj))) #use sliding_window get indices and patches
        labels = predict(patches, DEBUG) #predict zebra crossing for every patches 1 is zc 0 is background
        indices = np.array(indices)
        ret, location = getlocation(indices, labels, Ni, Nj)
        #draw
        # if DEBUG:
        #     for i, j in indices[labels == 1]:
        #         cv2.rectangle(img, (j, i), (j+Nj, i+Ni), (0, 0, 255), 3)
        if ret:
           cv2.rectangle(img, location[0], location[1], (255, 0, 255), 3)
        cv2.imshow("img", img)
        cv2.waitKey(1)
      

            









      
