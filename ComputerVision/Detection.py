

import cv2
import numpy as np

#EDGE DETECTION
image = cv2.imread('/Users/giorgiasavo/Documents/projects/personal/AI_python/ComputerVision/image_flower.jpg')
cv2.imwrite('edge_flowers.jpg', cv2.Canny(image, 200, 300))
cv2.imshow('edges', cv2.imread('edge_flowers.jpg'))
cv2.waitKey(1000)

#FACE DETECTION
face_detection = cv2.CascadeClassifier('/Users/giorgiasavo/Documents/projects/personal/AI_python/ComputerVision/haarcascade_frontalface_default.xml')
img = cv2.imread('/Users/giorgiasavo/Documents/projects/personal/AI_python/ComputerVision/face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting into gray scale to be accepted
faces = face_detection.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
cv2.imwrite('face_detected.jpg', img)

#EYE DETECTION
eye_cascade = cv2.CascadeClassifier("/Users/giorgiasavo/Documents/projects/personal/AI_python/ComputerVision/haarcascade_eye.xml")
img2 = cv2.imread('/Users/giorgiasavo/Documents/projects/personal/AI_python/ComputerVision/face2.jpg')
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray, 1.03, 5)
for(ex, ey, ew, eh) in eyes:
    img2 = cv2.rectangle(img2, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
cv2.imwrite('eye_detected.jpg', img2)



