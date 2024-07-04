"""
Explanation:

Background modeling with a Mixture of Gaussians (MOG) is a powerful technique 
in computer vision used for distinguishing moving objects from a static background 
in video sequences. This approach models the background by employing multiple 
Gaussian distributions to represent the pixel values. Unlike simpler methods 
that rely on a single Gaussian distribution, the MOG algorithm can adapt to 
variations in the background, such as changes in lighting, movement of tree 
branches, or subtle changes due to shadows. Each pixel in the frame is represented 
by a mixture of Gaussians, with each Gaussian component accounting for different 
color intensities and variations that might occur over time. 

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
