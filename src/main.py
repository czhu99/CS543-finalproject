import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt

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

#iqr outlier removal
def remove_outliers(data):
    data = np.sort(data)
    Q1 = np.percentile(data, 25, interpolation = 'midpoint') 
    Q2 = np.percentile(data, 50, interpolation = 'midpoint') 
    Q3 = np.percentile(data, 75, interpolation = 'midpoint') 
    IQR = Q3 - Q1
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR

    outliers = []
    for x in data:
        if ((x > up_lim) or (x < low_lim)):
            outliers.append(x)

    outliers = np.array(outliers)
    data = np.setdiff1d(data, outliers)

    return data

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

print('Inst mean: ' + str(np.mean(inst_ssds)))
print('Zhang mean: ' + str(np.mean(zhang_ssds)))
print('Inst median: ' + str(np.median(inst_ssds)))
print('Zhang median: ' + str(np.median(zhang_ssds)))

inst_ssds = remove_outliers(inst_ssds)
zhang_ssds = remove_outliers(zhang_ssds)

print('Inst mean (no outliers): ' + str(np.mean(inst_ssds)))
print('Zhang mean: (no outliers)' + str(np.mean(zhang_ssds)))
print('Inst median (no outliers): ' + str(np.median(inst_ssds)))
print('Zhang median: (no outliers)' + str(np.median(zhang_ssds)))

plt.style.use('seaborn-deep')
plt.xlabel('SSD Between Colorized and Original Images (1e8)')
plt.ylabel('Frequency')
plt.title('Comparison of SSDs: Instance-Aware Implementation vs. Zhang et al. (outliers removed)')

instlabel = 'Instance-Aware Implementation'
zhanglabel = 'Zhang et al\'s Implementation'
plt.hist([inst_ssds, zhang_ssds], label=[instlabel, zhanglabel])
plt.legend(loc='upper right')
plt.show()

