"""
Explanation:

Tracking moving objects in a video stream is a vital task in computer vision, 
enabling continuous monitoring and analysis of objects over time. Unlike simple 
detection, which only identifies objects in individual frames, tracking involves 
following these objects as they move across multiple frames. This process requires 
robust algorithms capable of maintaining the identity of the tracked objects despite 
changes in their appearance, occlusions, and varying lighting conditions. 

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
