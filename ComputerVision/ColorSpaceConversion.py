import cv2
image = cv2.imread('/Users/giorgiasavo/Documents/projects/personal/AI_python/ComputerVision/image_flower.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', image)
cv2.waitKey(5000)