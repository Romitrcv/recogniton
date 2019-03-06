# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:28:06 2019

@author: Romit Chand Verma
"""

#trying to mix face recog and max contour i.e single hand
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\\Users\\Romit Chand Verma\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml') 

lowerBound=np.array([5,20,70])
upperBound=np.array([19,255,255])
kernelOpen=np.ones((5,5))
cam=cv2.VideoCapture(0)
X=1
Y=1
Xn=1
Yn=1
while True:
    ret,img=cam.read()
    image = img
    
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    
#search face and put a rectangle of blue colour on face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #
    for (x,y,w,h) in faces:#
        #cv2.rectangle(output,(x,y),(x+w,y+h),(255,255,0),2) #
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        print(x,y,w,h)#
        X=x
        Y=y
        Xn=x+w
        Yn=y+h
    for i in range(X,Xn-2):
        for j in range(Y,Yn-2):
            mask[j][i]=0
    
    
    output = cv2.bitwise_and(image, image, mask=mask)

    ret,thresh = cv2.threshold(mask, 40, 255, 0)
    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
    # draw in blue the contours that were founded
      #  cv2.drawContours(output, contours, -1, 255, 0)

    #find the biggest area
        c = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
    
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)


 # show the images   
    cv2.imshow("Result", np.hstack([image, output]))
    cv2.imshow("mask",mask)

    cv2.waitKey(10)

