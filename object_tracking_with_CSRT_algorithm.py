"""
Explanation:

The CSRT (Channel and Spatial Reliability Tracking) algorithm is a highly 
accurate and robust method for object tracking, specifically designed to handle 
challenging scenarios where other trackers might fail. CSRT builds upon the 
concept of correlation filters but introduces significant improvements to enhance 
both accuracy and robustness. One of the key strengths of CSRT is its ability to 
incorporate spatial reliability maps, which weigh different parts of the tracked 
object based on their reliability. This means that areas of the object that are 
more consistent and stable over time are given more importance, while less reliable 
areas are down-weighted. 

"""

import cv2

tracker = cv2.TrackerCSRT_create()
cap = cv2.VideoCapture('video.mp4')
ret, frame = cap.read()
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    ret, bbox = tracker.update(frame)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
