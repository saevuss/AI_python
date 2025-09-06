import cv2
image = cv2.imread('/Users/giorgiasavo/Documents/projects/personal/AI_python/ComputerVision/image_flower.jpg')
#showing the image
cv2.imshow('image_flower', image)
cv2.imwrite('image_flower.png', image)
cv2.destroyAllWindows()