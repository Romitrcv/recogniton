# OpenCV program to detect face in real time 
# import libraries of python OpenCV 
# where its functionality resides 


# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalface_default.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:28:06 2019
@author: Romit Chand Verma
"""

#trying to mix face recog and max contour i.e single hand
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\\Users\\DELL\\Anaconda\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml') 

lowerBound=np.array([0,48,80])
upperBound=np.array([20, 255, 255])
#lowerBound=np.array([5,20,70])
#upperBound=np.array([19,255,255])
cen_h2_x=cen_h2_y=cen_h1_x=cen_h1_y=cen_face_x=cen_face_y=0
kernelOpen=np.ones((5,5))
cam=cv2.VideoCapture(0)
X=1
Y=1
Xn=1
Yn=1
areaArray = []
while True:
    ret,img=cam.read()
    
    image = img
    
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    
#perform dilation on mask image     
    kernel = np.ones((4,4), np.uint8)
    #img_er=cv2.erode(mask, kernel, iterations=1)
    img_dil=cv2.dilate(mask, kernel, iterations=2)
    kernel = np.ones((2,2), np.uint8)
    img_erose=cv2.erode(img_dil, kernel, iterations=2)
    #cv2.imshow('erose', img_er)
    #cv2.imshow('dilate',img_dil)
   # cv2.imshow('dilate+erose',img_erose)
    
    
    mask= img_erose
#search face and put a rectangle of blue colour on face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #
    for (x,y,w,h) in faces:#
        #cv2.rectangle(output,(x,y),(x+w,y+h),(255,255,0),2) #
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.circle(img,(int(x+w/2),int(y+w/2)), 5, (0,0,255), 3)
        cen_face_x=(int(x+w/2))
        cen_face_y=(int(y+w/2))
      #  print(x,y,w,h)#
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
    try:
        
        if len(contours) != 0:
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                areaArray.append(area)
                #first sort the array by area
        #    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
                
                # draw in blue the contours that were founded
                #  cv2.drawContours(output, contours, -1, 255, 0)
                
                #find the biggest area
            sorteddata= sorted(contours,key=cv2.contourArea)
            c = max(contours, key = cv2.contourArea)
                
            x,y,w,h = cv2.boundingRect(c)        
                # draw the first rectangle (in green)    
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(img,(int(x+w/2),int(y+w/2)), 5, (0,0,255), 3)
            cen_h1_x=(int(x+w/2))
            cen_h1_y=(int(y+w/2))
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                #find the nth largest contour [n-1][1], in this case 2
 #           first = sorteddata[0][1]
            #second = sorteddata[1][1]       
            second=sorteddata[-2]
  #          x,y,w,h = cv2.boundingRect(first)        
        # draw the first rectangle (in green)    
   #     cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
    #    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

            x,y,w,h = cv2.boundingRect(second)        
    # draw the first rectangle (in green)    
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(img,(int(x+w/2),int(y+w/2)), 5, (0,0,255), 3)
            cen_h2_x=(int(x+w/2))
            cen_h2_y=(int(y+w/2))
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    except:
        print("limb not found")

 # show the images   
    cv2.imshow("Result", np.hstack([image, output]))
    cv2.imshow("mask",mask)
    print "\n\ncentroids of \nface \nx=",cen_face_x,"\ty=",cen_face_y 
    print "\ncentroids of \nhand 1 \nx=",cen_h1_x,"\ty=",cen_h1_y 
    print "\ncentroids of \nhand 2 \nx=",cen_h2_x,"\ty=",cen_h2_y 
    
    cv2.waitKey(10)
