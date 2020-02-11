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
# plt.imshow(not_a)
# plt.show()
new_img = cv2.bitwise_and(new_img, new_img, mask=a)
new_img = cv2.add(new_img, not_a)

cv2.imwrite("image_wb.jpg", new_img)


def noise(noise_typ, slika):

    if noise_typ == "gauss":                                # gauss
        row, col, ch = slika.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.8
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = slika + gauss
        cv2.imwrite("image_gauss.jpg", noisy)
        return noisy

    elif noise_typ == "s&p":                             # salt & pepper
        # row, col, ch = slika.shape
        s_vs_p = 0.5
        amount = 0.004*5
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
        noisy = out
        cv2.imwrite("image_s&p.jpg", noisy)
        return noisy

    elif noise_typ == "poisson":                        # pisson
        val = len(np.unique(slika))
        val = 2 ** np.ceil(np.log2(val))
        noisy = np.random.poisson(slika * val) / float(val)
        cv2.imwrite("image_poisson.jpg", noisy)
        return noisy

    elif noise_typ == "speckle":                        # speckle
        row, col, ch = slika.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = slika + slika * gauss
        cv2.imwrite("image_speckle.jpg", noisy)
        return noisy


image = cv2.imread("image_wb.jpg")
noise("gauss", image)
noise("poisson", image)
noise("speckle", image)
noise("s&p", image)
