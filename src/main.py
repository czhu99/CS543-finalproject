import cv2 
import os
import numpy as np

orig_path = '../data/gatto_resize/'
grayscale_path = '../data/gray_cat/'
instance_path = '../data/inst_cat/'
zhang_path = '../data/zhang_resize/'

#function taken from https://gist.github.com/nimpy/54ccb199c978a5074cdcd35fc696a904
def calculate_ssd(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)

inst_ssds = []
zhang_ssds = []

for i in range(1668):
    img1 = cv2.imread(orig_path + 'cat' + str(i) + '.jpg')
    img2 = cv2.imread(instance_path + 'cat' + str(i) + '_inst.png')
    img3 = cv2.imread(zhang_path + 'cat' + str(i) + '_zhang.png')

    inst_ssd = calculate_ssd(img1, img2)
    zhang_ssd = calculate_ssd(img1, img3)
    inst_ssds.append(inst_ssd)
    zhang_ssds.append(zhang_ssd)

    # print('Inst ' + str(i) + ' ' + str(inst_ssd))
    # print('Zhang ' + str(i) + ' ' + str(zhang_ssd))

print('Inst mean:' + str(np.median(inst_ssds)))
print('Zhang mean:' + str(np.median(zhang_ssds)))