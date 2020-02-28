# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:33:12 2020

@author: ext_AtKs1
"""

import cv2
import numpy as np
from datetime import datetime

cam = cv2.VideoCapture(0)       # Captures the video from webcam
ret, frame = cam.read() 
r = cv2.selectROI(frame)    

cv2.namedWindow("test1")

cv2.namedWindow("test2")

img_counter = 0
hue = 0
val = 0
sat = 0
while True:
    ret, frame = cam.read()     # reads the images from video frames
    imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]  # Crop image
    hsv_frame = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)
    
    # Put current DateTime on each frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,str(datetime.now()),(10,30), font, 1,(0,0,255),2,cv2.LINE_AA)
    # Display the image
    cv2.imshow('a',imCrop)
    
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
    
    gray_frame = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)
    ret,thresh1 = cv2.threshold(gray_frame,200,255,cv2.THRESH_BINARY)  
    
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # returns boundary points of the contour
    
    if np.size(contours) > 0:
        
        #print("LED found")
        contour_sizes = []
        for contour in contours:
            contour_sizes.append(cv2.contourArea(contour))
            
        ind = np.argmax(contour_sizes)
        x,y,w,h = cv2.boundingRect(contours[ind])   # in order to form the bounding rectangle
        
        crop_img = frame[y:y+h, x:x+w]
        
        if (cv2.contourArea(contour) > 100):
            cv2.rectangle(frame,(r[0]+x,r[1]+y),(r[0]+x+w,r[1]+y+h),(0,0,255),2)    # draws a rectangle around area of interest in the frame
        
            a = [img_hsv[l,m,0] for [m,l] in contours[ind][:,0,:]]                                   # list of hue values of the boundary points of the contour
            s = [img_hsv[l,m,1] for [m,l] in contours[ind][:,0,:]]                                   # returns saturation value of the boundary points 
            v = [img_hsv[l,m,2] for [m,l] in contours[ind][:,0,:]] 
#          print(np.amax(b),np.shape(img_hsv))
          
            for i in range(np.size(a)):
                if a[i]>170:
                    a[i]=a[i]-180                                                 # Hue value of red if 175, should consider as -5
            hue=np.mean(a)                                                        # average hue values of the boundary points
            sat=np.mean(s)                                                        # average saturation values of the boundary points
            val=np.mean(v)
            if(sat<30):
                print("LED Off")
            else:
                print("LED On")
                if (hue<115 and hue>95):
                    print("BLUE")
                elif (hue<75 and hue>63):
                    print("YELLOW")
                elif (hue<60 and hue>48):
                    print("WHITE")
                
    cv2.rectangle(frame,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),2)
    print(hue,sat,val)
    cv2.imshow('test2',thresh1)
    cv2.imshow("test1", frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()