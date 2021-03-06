import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

img_path = r"D:\Neural Music Sheets\data"  # novi dir
os.chdir(img_path)

# saving image into a white bg
img = cv2.imread("training.png", cv2.IMREAD_UNCHANGED)
# plt.imshow(img)
# plt.show()
b, g, r, a = cv2.split(img)
print(img.shape)

new_img = cv2.merge((b, g, r))
not_a = cv2.bitwise_not(a)
not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)
new_img = cv2.bitwise_and(new_img, new_img, mask=a)
new_img = cv2.add(new_img, not_a)

cv2.imwrite("image_wb.jpg", new_img)

img = cv2.imread('image_wb.jpg',0)
rows,cols = img.shape

warps=[]                    # lista warpova
for k in range (0, 10):
    M = np.float32([[1,0,(k*2)-5],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))     # offsetiranje
    warps.append(dst)

warp_speck = []
j=0
for i in range(len(warps)):
    slika = warps[i]
    name = "warp" + str(j) + ".jpg"
    name2 = "warp_speck" + str(j) + ".jpg"
    name3 = "s&p_speck" + str(j) + ".jpg"
    cv2.imwrite("resources/" + name, warps[i])

    row, col = slika.shape
    gauss = np.random.randn(row, col)
    gauss = gauss.reshape(row, col)
    noisy = slika + slika * gauss
    cv2.imwrite("resources/" + name2, noisy)

    s_vs_p = 0.5
    amount = 0.004 * 5
    out = np.copy(slika)

    # Salt mode
    num_salt = np.ceil(amount * slika.size * s_vs_p)
    coord = [np.random.randint(0, i - 1, int(num_salt))
             for i in slika.shape]
    out[coord] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * slika.size * (1. - s_vs_p))
    coord = [np.random.randint(0, i - 1, int(num_pepper))
             for i in slika.shape]
    out[coord] = 0
    noisy2 = out
    cv2.imwrite("resources/" + name3, noisy2)

    j = j + 1

    
    
from keras.applications.vgg16 import VGG16
model = VGG16()
