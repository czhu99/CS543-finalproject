import cv2 
img = cv2.imread("cat0.jpg")
resized_image = cv2.resize(img, (256, 256))
cv2.imwrite("cat0_resize.jpg", resized_image)