import cv2 
import os

data_dir = "../data/gatto/"
output_dir = "../data/gray_cat/"

i = 0
for file in os.listdir(data_dir):
    os.rename(data_dir + file, data_dir + "cat" + str(i) + '.jpg')
    i += 1

for file in os.listdir(data_dir):
    img = cv2.imread(data_dir + file)
    num = file.split('.')[0]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_dir + num + '_gray.jpg', gray_img)

# for file in os.listdir(data_dir):
#     if "gray" in file:
#         os.remove(data_dir + file)
