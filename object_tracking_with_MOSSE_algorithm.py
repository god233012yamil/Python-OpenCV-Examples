"""
Explanation:

The MOSSE (Minimum Output Sum of Squared Error) algorithm is a fast and 
efficient method for object tracking, known for its high performance in 
real-time applications. Developed by David S. Bolme and colleagues, MOSSE 
is based on the concept of adaptive correlation filters, which are optimized 
to produce high response peaks at the target object's location. One of the 
standout features of MOSSE is its ability to maintain high tracking accuracy 
while operating at impressive speeds, making it ideal for scenarios that 
require real-time processing, such as live video feeds and interactive systems. 

"""

import cv2

tracker = cv2.TrackerMOSSE_create()
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

