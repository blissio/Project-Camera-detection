import cv2

import torch

import numpy as np

import time

cap = cv2.VideoCapture(0)


kernel = np.ones((5, 5), np.uint8) 

lower_red1 = np.array([0, 150, 120])
upper_red1 = np.array([8, 255, 255])

lower_red2 = np.array([172, 150, 120])
upper_red2 = np.array([180, 255, 255])

MAX_DISPLACEMENT = 30 

prev_z = None

# Time
dt = 1.0 / 30.0

last_time = None


class KalmanFiltering:
    def __init__(self):
        self.x = np.zeros((4, 1), dtype=np.float32)

        self.P = np.eye(4, dtype=np.float32) * 1000

        self.Q = np.eye(4, dtype=np.float32) * 0.03

        self.R = np.eye(2, dtype=np.float32) * 10

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.I = np.eye(4, dtype=np.float32)

        self.initialized = False

    def predict(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.x = F @ self.x

        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy()
    
    def update(self, z):
        if z.shape != (2,1):
            z = z.reshape(2, 1)
        
        y = z - (self.H @ self.x)


        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I_KH = self.I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()
    
    def initialize(self, measurement): 
        if measurement.shape != (2, 1):
            measurement = measurement.reshape(2, 1)
        
        self.x[0:2] = measurement
        self.x[2:4] = 0

        self.initialized = True

kf = KalmanFiltering()
while True:
    ret,frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if last_time is not None:
        dt = current_time - last_time
    else:
        dt = 1/30

    last_time = current_time



    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if kf.initialized:
        prediction = kf.predict(dt)
        predicted_pos = (int(prediction[0, 0]), int(prediction[1, 0]))
        
        cv2.circle(frame, predicted_pos, 7, (255, 0, 0), 2)
        
        vx, vy = prediction[2, 0], prediction[3, 0]
        speed = np.sqrt(vx**2 + vy**2)
        cv2.putText(frame, f"Speed: {speed:.1f} px/s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"dt: {dt*1000:.1f} ms", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {1/dt:.1f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            x, y, w, h = cv2.boundingRect(cnt)

            z_t = np.array([[x + w/2], [y + h/2]], dtype=np.float32)
            
            if prev_z is not None:
                displacement = np.linalg.norm(z_t.flatten() - prev_z)
                if displacement > 100:
                    continue
            
            prev_z = z_t.flatten()
            
            if not kf.initialized:
                kf.initialize(z_t)
            
            estimated = kf.update(z_t)
            estimated_pos = (int(estimated[0, 0]), int(estimated[1, 0]))
            
            center_raw = (int(z_t[0, 0]), int(z_t[1, 0]))
            cv2.circle(frame, center_raw, 5, (0, 0, 255), -1)
            
            cv2.circle(frame, estimated_pos, 5, (0, 255, 0), -1)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.putText(frame, "Raw", center_raw, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Filtered", estimated_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #if kf.initialized:
             #   cv2.putText(frame, "Predicted", predicted_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    #for cnt in contours:
        
     #   if cv2.contourArea(cnt) > 200:
      #      x, y, w, h = cv2.boundingRect(cnt)


       #     z_t = np.array([x + w/2, y + h/2], dtype=np.float32)
        #    if prev_z is not None:
         #       displacement = np.linalg.norm(z_t - prev_z)
          #      if displacement > 100:
           #         continue
            #prev_z = z_t
            #center = (int(z_t[0]), int(z_t[1]))
            #cv2.circle(frame, center, 5, (0, 0, 255), -1)

         #   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    cv2.imshow("Color Tracking", frame)
    cv2.imshow("Mask", mask)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()