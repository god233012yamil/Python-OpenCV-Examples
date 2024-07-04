"""
Explanation:

Motion detection with frame differencing is a straightforward yet effective 
method for identifying moving objects in video sequences. This technique operates 
by comparing consecutive frames in a video stream to detect changes that signify 
motion. Frame differencing involves subtracting the pixel values of one frame from 
the corresponding pixel values of the previous frame. The resulting difference 
image highlights the areas where significant changes have occurred, effectively 
isolating the moving objects from the static background. This method is particularly 
advantageous due to its simplicity and computational efficiency, making it suitable 
for real-time applications where quick detection is essential.

"""

import cv2

cap = cv2.VideoCapture('video.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 700:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Frame', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    if not ret or cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
