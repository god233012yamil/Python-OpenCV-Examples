"""
Explanation:

Motion detection using optical flow is a sophisticated technique in computer vision that 
analyzes the apparent motion of objects between consecutive frames. Optical flow calculates 
the motion of each pixel by examining the changes in intensity patterns, which allows us to 
detect and track moving objects with high accuracy. Unlike simpler motion detection methods 
that rely on frame differencing or background subtraction, optical flow provides a more 
detailed and continuous representation of motion. It works by computing a dense field of 
displacement vectors, which represent the motion of points from one frame to the next. 

"""

import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Frame', bgr)
    if cv2.waitKey(30) & 0xFF == 27:
        break
    prvs = next

cap.release()
cv2.destroyAllWindows()
