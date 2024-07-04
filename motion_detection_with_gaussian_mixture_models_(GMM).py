"""
Explanation:

Motion detection with Gaussian Mixture Models (GMM) is a sophisticated and 
highly effective technique for identifying moving objects in video streams. 
This method stands out for its ability to model the background dynamically, 
adapting to changes over time, which makes it ideal for complex environments 
where lighting conditions and background elements can vary. The core idea 
behind GMM is to represent each pixel in the video frame as a mixture of 
several Gaussian distributions. Each Gaussian component accounts for different 
aspects of the pixel's color and intensity, allowing the model to capture a 
wide range of background variations.

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
