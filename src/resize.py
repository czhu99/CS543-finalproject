import cv2 
import os

img = cv2.imread("cat0.jpg")
resized_image = cv2.resize(img, (256, 256))
cv2.imwrite("cat0_resize.jpg", resized_image)

orig_path = '../data/gatto/'
orig_resize = '../data/gatto_resize/'

zhang_path = '../data/zhang_cat/'
zhang_resize = '../data/zhang_resize/'

for file in os.listdir(orig_path):
    img = cv2.imread(orig_path + file)
    resized_image = cv2.resize(img, (256, 256))
    cv2.imwrite(orig_resize + file, resized_image)

for file in os.listdir(zhang_path):
    img = cv2.imread(zhang_path + file)
    resized_image = cv2.resize(img, (256, 256))
    cv2.imwrite(zhang_resize + file, resized_image)
