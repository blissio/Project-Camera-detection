import cv2

import torch

import numpy as np

cap = cv2.VideoCapture(0)


kernel = np.ones((5, 5), np.uint8) 

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])



while True:
    ret,frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    #for cnt in contours:
     #   if cv2.contourArea(cnt) > 200:
      #      epsilon = 0.02
       #     approx = cv2.approxPolyDP(cnt, epsilon, True)
        #    cv2.drawContours(frame, [approx], 0,(0,255,0), 2)
        #x, y, w, h = cv2.boundingRect(cnt)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 4)
    
    cv2.imshow("Color Tracking", frame)
    cv2.imshow("Mask", mask)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()