"""
Explanation:

The CAMSHIFT (Continuously Adaptive Mean Shift) algorithm is a robust and adaptive 
method for tracking objects in video sequences, building on the foundational Mean 
Shift algorithm. CAMSHIFT is particularly effective for tracking objects that change 
in size, rotation, and appearance over time, making it well-suited for dynamic and 
real-world applications. The algorithm begins with a target model, often defined by 
a histogram of colors, which represents the object to be tracked. As the video 
progresses, CAMSHIFT continuously updates this model to adapt to changes in the 
object's appearance and position. 

"""

import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
ret, frame = cap.read()
x, y, w, h = cv2.selectROI(frame, False)
track_window = (x, y, w, h)
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    cv2.imshow('Tracking', img2)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
