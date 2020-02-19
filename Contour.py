# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:33:12 2020

@author: ext_AtKs1
"""

import cv2
import numpy as np

cam = cv2.VideoCapture(0)       # Captures the video from webcam

cv2.namedWindow("test1")

cv2.namedWindow("test2")

img_counter = 0

while True:
    ret, frame = cam.read()     # reads the images from video frames
    
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
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    low_red1 = np.array([0, 155, 84])
    high_red1 = np.array([10, 255, 255])
    kernel = np.ones((5,5),np.uint8)
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)                   #creates a b/w image and only keeps the pixels within this hsv range 
    red_mask1 = cv2.inRange(hsv_frame, low_red1, high_red1)
    red_mask = cv2.bitwise_or(red_mask,red_mask1)
    
     # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
   #blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    
    # Green color
    low_green = np.array([40, 100,20])
    high_green = np.array([70, 255,255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    
     # Yellow color
    low_yellow = np.array([240, 197, 110])
    high_yellow = np.array([200, 255, 0])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    
    
    
    contours_red, hierarchy = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # returns boundary points of the contour
    contours_blue, hierarchy = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, hierarchy = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    
  #  contour_sizes_red = [(cv2.contourArea(contour), contour) for contour in contours_red]  
   

        
  #  print(contour_sizes_red)
    
    
    if np.size(contours_red) > 0:
        
        print("Red")
        contour_sizes_red = []
        for contour in contours_red:
            contour_sizes_red.append(cv2.contourArea(contour))
            
        ind = np.argmax(contour_sizes_red)
        x,y,w,h = cv2.boundingRect(contours_red[ind])       # in order to form the bounding rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)    # draws a rectangle around area of interest in the frame
       
    if np.size(contours_blue) > 0:
        
        print("blue")
        contour_sizes_blue = []
        for contour in contours_blue:
            contour_sizes_blue.append(cv2.contourArea(contour))
                
        ind = np.argmax(contour_sizes_blue)
        x,y,w,h = cv2.boundingRect(contours_blue[ind])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    if np.size(contours_green) > 0:       # If contour found
        
        print("green")
        contour_sizes_green = []
        for contour in contours_green:  
            contour_sizes_green.append(cv2.contourArea(contour))
                
        ind = np.argmax(contour_sizes_green)
        x,y,w,h = cv2.boundingRect(contours_green[ind])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)   
        
    if np.size(contours_yellow) > 0:
        
        print("yellow")
        contour_sizes_yellow = []
        for contour in contours_yellow:
            contour_sizes_yellow.append(cv2.contourArea(contour))
                
        ind = np.argmax(contour_sizes_yellow)
        x,y,w,h = cv2.boundingRect(contours_yellow[ind])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 255, 0),2)
        
        
        
#        print(np.shape(contours_red))
         # draws a rectangle around LED 
    
    cv2.imshow('test2',red_mask)
    cv2.imshow("test1", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()

cv2.destroyAllWindows()