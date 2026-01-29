import cv2

import torch

import numpy as np

cap = cv2.VideoCapture(0)


kernel = np.ones((5, 5), np.uint8) 

lower_red1 = np.array([0, 150, 120])
upper_red1 = np.array([8, 255, 255])

lower_red2 = np.array([172, 150, 120])
upper_red2 = np.array([180, 255, 255])


MAX_DISPLACEMENT = 30 

prev_z = None

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


            z_t = np.array([x + w/2, y + h/2], dtype=np.float32)
            if prev_z is not None:
                displacement = np.linalg.norm(z_t - prev_z)
                if displacement > 100:
                    continue
            prev_z = z_t
            center = (int(z_t[0]), int(z_t[1]))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    cv2.imshow("Color Tracking", frame)
    cv2.imshow("Mask", mask)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()