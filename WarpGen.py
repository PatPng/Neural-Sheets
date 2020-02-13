import numpy as np
import cv2
import os

path = ".\\resources\\img02\\"

images = []
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            if 'el' in file:
                images.append(file)

for i in images:
    for k in range(0, 5):
        img = cv2.imread(path + i, 0)
        rows, cols = img.shape
        M = np.float32([[1, 0, k - 2], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, rows))  # offsetiranje
        element_name = "warp" + str(i[:-4]) + str(k) + ".jpg"
        cv2.imwrite("resources/dataset/" + element_name, dst)
