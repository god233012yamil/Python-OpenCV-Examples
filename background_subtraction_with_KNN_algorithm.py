"""
Explanation:

Background subtraction with the K-Nearest Neighbors (KNN) algorithm is a 
robust technique for detecting moving objects in video streams, particularly 
in environments with varying illumination and dynamic backgrounds. The KNN 
algorithm models the background by treating each pixel value as a sample in 
a higher-dimensional feature space, where it evaluates the similarity of new 
pixel values to a set of recent observations. By considering the K most similar 
samples, the algorithm classifies each pixel as either foreground or background 
based on its distance to these neighbors.

"""

import cv2

cap = cv2.VideoCapture('video.mp4')
fgbg = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    cv2.imshow('Frame', fgmask)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
