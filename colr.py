# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
 
lowerBound=np.array([5,20,70])
upperBound=np.array([19,255,255])

cam=cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))
while True:
    ret, img=cam.read()
        
    scale_percent = 60
    width =340 
    """ int(img.shape[1] * scale_percent / 100)"""
    height = 220 
    """int(img.shape[0] * scale_percent / 100)"""
    dim = (width, height)
    img=cv2.resize(img,dim)
    
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelOpen)
    cv2.imshow("maskclosed",maskClose)
    cv2.imshow("mask",mask)
    cv2.imshow("cam",img)

    #cv2.imshow("Result", np.hstack([img,mask]))
    cv2.waitKey(10)
    