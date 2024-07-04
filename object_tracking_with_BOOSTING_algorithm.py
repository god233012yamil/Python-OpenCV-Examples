"""
Explanation:

The BOOSTING algorithm is a robust and reliable method for object tracking, 
leveraging the power of ensemble learning to enhance tracking accuracy and 
resilience. At its core, BOOSTING uses a collection of weak classifiers, 
which are simple decision rules that individually may not perform well but 
collectively form a strong classifier. This ensemble approach allows BOOSTING 
to effectively handle variations in the appearance of the tracked object, 
making it well-suited for dynamic environments where objects may undergo 
changes in shape, color, and size. Each weak classifier focuses on different 
aspects of the object's appearance, and their combined decisions result in 
a more accurate and reliable tracking performance. 

"""

import cv2

tracker = cv2.TrackerBoosting_create()
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
