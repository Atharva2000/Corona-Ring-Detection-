# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:22:53 2020

@author: ext_AtKs1
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)
 
while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    kernel = np.ones((5,5),np.uint8)
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    dilation = cv2.dilate(red_mask,kernel,iterations = 100)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    
    # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Red", red)
    cv2.imshow("Blue", blue)
    cv2.imshow("Green", green)
    cv2.imshow("Result", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

upper_left = (50, 50)
bottom_right = (300, 300)
while True:
    _, image_frame = cap.read()
    
    #Rectangle marker
    r = cv2.rectangle(image_frame, upper_left, bottom_right, (100, 50, 200), 5)
    rect_img = image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    
    sketcher_rect = rect_img
    