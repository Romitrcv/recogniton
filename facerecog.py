# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:30:30 2019

@author: Romit Chand Verma
"""

# OpenCV program to detect face in real time 
# import libraries of python OpenCV 
# where its functionality resides 
import cv2 

# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalface_default.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 
face_cascade = cv2.CascadeClassifier('C:\\Users\\Romit Chand Verma\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml') 




# capture frames from a camera 
cap = cv2.VideoCapture(0) 

# loop runs if capturing has been initialized. 
while 1: 

	# reads frames from a camera 
	ret, img = cap.read() 

	# convert to gray scale of each frames 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

	# Detects faces of different sizes in the input image 
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

	for (x,y,w,h) in faces: 
		# To draw a rectangle in a face 
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
    #    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        #cv2.circle(img,(x,y),x,(255,255,0),2) 
#        cv2.circle(img, (x,y), 20, (222,225,0), 3)
        #cv2.circle(img,(447,63), 63, (0,0,255), -1)
		cv2.circle(img,(int(x+w/2),int(y+w/2)), 5, (255,0,0), 3)
	#	roi_gray = gray[y:y+h, x:x+w] 
	#	roi_color = img[y:y+h, x:x+w] 
 
	cv2.imshow('img',img) 

	# Wait for Esc key to stop 
	k = cv2.waitKey(30) & 0xff
	if k == 27: 
		break

# Close the window 
cap.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 
