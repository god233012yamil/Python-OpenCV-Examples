"""
Explanation:

Background subtraction is a fundamental technique in computer vision used to distinguish 
moving objects from the static background in a video stream. This method is pivotal for 
various applications, such as surveillance, traffic monitoring, and human-computer interaction, 
where detecting motion is crucial. The essence of background subtraction lies in its ability 
to model the background of a scene and identify significant changes that indicate moving objects. 
It begins by capturing a static background model, which can be updated over time to accommodate 
gradual changes like lighting variations or moving shadows.

"""

import cv2

cap = cv2.VideoCapture('video.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

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