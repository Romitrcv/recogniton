# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 01:05:08 2019

@author: Romit Chand Verma
"""
#finding largest contour
import numpy as np
import cv2

# load the image

cam=cv2.VideoCapture(0)
while True:
    ret,img=cam.read()
    image = img
# red color boundaries (R,B and G)

    lowerBound=np.array([5,20,70])
    upperBound=np.array([19,255,255])

# create NumPy arrays from the boundaries
#lower = np.array(lower, dtype="uint8")
#upper = np.array(upper, dtype="uint8")

# find the colors within the specified boundaries and apply
# the mask
#mask = cv2.inRange(image, lower, upper)
 #scale_percent = 60
#    width =340 
 #   """ int(img.shape[1] * scale_percent / 100)"""
#    height = 220 
#    """int(img.shape[0] * scale_percent / 100)"""
#    dim = (width, height)
#    img=cv2.resize(img,dim)
    
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
   
    output = cv2.bitwise_and(image, image, mask=mask)

    ret,thresh = cv2.threshold(mask, 40, 255, 0)
    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
    # draw in blue the contours that were founded
        cv2.drawContours(output, contours, -1, 255, 3)

    #find the biggest area
        c = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

# show the images
    cv2.imshow("Result", np.hstack([image, output]))

    cv2.waitKey(10)