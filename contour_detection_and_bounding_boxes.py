"""
Explanation:

Contour detection helps to find the boundaries of objects within a frame. By drawing 
bounding boxes around these contours, we can highlight the detected objects. This method 
is particularly useful in various applications, from object detection and tracking in 
video surveillance to shape analysis and recognition in industrial automation. We can 
not only visualize the detected objects but also facilitate further processing, such as 
object classification and behavior analysis.

"""

import cv2

cap = cv2.VideoCapture('video.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()