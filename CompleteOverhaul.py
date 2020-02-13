import cv2
import numpy as np
import os
# TODO - make it into a function that has an arg = path(str)
file_name = "img02.jpg"                                                         # image Name
img = cv2.imread(file_name)                                                     # read

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                         # transform into gray img
th, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)     # pixels have 1 of 2 values
non_zero_elements = cv2.findNonZero(thr)                                             # find the non zero pixels
non_zero_min_area = cv2.minAreaRect(non_zero_elements)                               # mark the area with a rectangle

(cx, cy), (w, h), ang = non_zero_min_area                                            # find rotated matrix
M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
rotated = cv2.warpAffine(thr, M, (img.shape[1], img.shape[0]))                       # do the rotation

# expand to fit A4 format
bordered = cv2.copyMakeBorder(rotated, 0, int((w*1.414)-h), 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)                            # reduce matrix to a vector

th = 2                                                                               # change the threshold (empirical)
H, W = img.shape[:2]                                                                 # picture dimensions

upperBound = [y for y in range(H - 1) if hist[y] <= th < hist[y + 1]]                # upper bounds
lowerBound = [y for y in range(H - 1) if hist[y] > th >= hist[y + 1]]                # lower bounds

up_array = np.asarray(upperBound)                                                    # list to array conversion
up = (H - up_array)
low_array = np.asarray(lowerBound)
low = (H - low_array)

slices = []
for i in range(len(up_array)):                                                     # row slicing
    if(low_array[i] + 1) + int(H/350) < H and up_array[i] - int(H / 70) > 0:       # expand the slice vertically
        h_slice = bordered[up_array[i] - int(H / 70):(low_array[i] + 1) + int(H / 350), 0:W]
    else:                                                                          # don't expand on the edges
        h_slice = bordered[up_array[i]:(low_array[i] + 1), 0:W]
    slices.append(h_slice)                                                         # save all the slices in a list

# on standard A4 paper(21 cm height), 1 note line is 1 cm -> 1/21 ~= 0.04, so ignore smaller slices
slices[:] = [s for s in slices if not len(s) < 0.04 * H]                           # remove unwanted slices

valid_slices_pixel_mean = []                                                       # find the mean value of each slice
for s in slices:
    valid_slices_pixel_mean.append(np.mean(s))
mean = np.mean(valid_slices_pixel_mean)                                            # find the mean value of all slices

j = 0
for i in range(len(slices)):                                                        # save the valid slices
    # wanted slices have approximately the same mean of pixels, ignore the unwanted lines(+- 15% of mean)
    if 1.30 * mean > valid_slices_pixel_mean[i] > 0.70 * mean:
        sliceName = "slice" + str(j) + ".jpg"                                       # slice naming
        path = "resources/" + file_name[:-4] + "/"                             # directory for the slices
        try:                                                                        # create the dir if it doesn't exist
            os.makedirs(path)
        except FileExistsError:
            pass
        cv2.imwrite(path + sliceName, slices[i])                                    # save the slices in that directory
        j = j + 1                                                                   # name slices iteratively




def crop_image(img):

    h, w = img.shape[:2]  # image dimensions
    img_mean = np.mean(img)  # find the mean of the image

    for i in range(len(img[0])):  # go horizontally; len(img[0]) = no. of columns
        column_mean = 0  # calculate the mean of every column
        for j in range(int(len(img) / 2), int(3 * len(img) / 4), 1):  # star at the middle of a picture
            column_mean = column_mean + np.mean(img[j][i])  # add the means of all the pixels in a column
        column_mean = column_mean / (len(img) / 2)  # divide by the number of elements(rows) in column
        if column_mean > img_mean:  # cut away the spaces before score lines
            img = img[0:h, i:len(img[0])]  # crop the image
            break  # break when done

    for i in range(len(img[0]) - 1, 0, -1):  # go backwards (end to 0, with step being -1)
        column_mean = 0  # calculate the mean of every column
        for j in range(len(img)):
            column_mean = column_mean + np.mean(img[j][i])  # add the means of all the pixels in a column
        column_mean = column_mean / len(img)  # divide by the number of elements(rows) in column
        if column_mean > img_mean:  # cut away the spaces after score lines
            img = img[0:h, 0:i]  # crop the image
            break  # break when done
    return img


def erode_dilate(img):
    kernel1 = np.ones((2, 2), np.uint8)  # kernel with dimensions 2x2, 1 if all = 1
    eroded_img = cv2.erode(img, kernel1, iterations=1)  # first erode
    dilated_img = cv2.dilate(eroded_img, None, iterations=1)  # then dilate
    return dilated_img


def find_histogram(dilated_img):
    min_mean = 10000  # result of the minimum mean of a vertical line

    for i in range(len(dilated_img[0])):  # go horizontally; len(img[0]) = no. of columns
        column_mean = 0  # calculate the mean of every column
        for j in range(len(dilated_img)):
            column_mean = column_mean + np.mean(dilated_img[j][i])  # add the means of all the pixels in a column
        column_mean = column_mean / len(dilated_img)  # divide by the number of elements(rows) in column
        if column_mean < min_mean:  # if it is smaller than the minimum...
            min_mean = column_mean  # ... make it a new minimum

    hist = [0]  # histogram of the picture
    for i in range(len(dilated_img[0])):  # go horizontally; len(img[0]) = no. of columns
        column_mean = 0  # calculate the mean of every column
        for j in range(len(dilated_img)):
            column_mean = column_mean + np.mean(dilated_img[j][i])  # add the means of all the pixels in a column
        column_mean = column_mean / len(dilated_img)  # divide by the number of elements(rows) in column

        if column_mean > 1.1 * min_mean:  # put 1 in a histogram if a line is not empty
            hist.append(1)
        else:  # put 0 in a histogram if a line is empty
            hist.append(0)
    return hist


def get_element_coordinates(dilated_img, hist):
    x_cut_start = []  # coordinates for the left side of the element
    x_cut_end = []  # coordinates for the right side of the element
    for i in range(len(hist)):  # find the edges (rising and falling edge)
        if i > 0:
            if hist[i - 1] == 0 and hist[i] == 1:  # find the starting x coordinate(rising edge)
                x_cut_start.append(i - 1)
            elif hist[i - 1] == 1 and hist[i] == 0:  # find the starting x coordinate(falling edge)
                x_cut_end.append(i - 1)
    x_cut_end.append(len(dilated_img[0] - 1))  # last coordinate is the end of the picture
    return x_cut_start, x_cut_end


def get_elements_from_image(path, x_cut_start, x_cut_end, img, element_number):
    large_elements = []
    large_elements_index = []
    h, w = img.shape[:2]  # image dimensions
    for i in range(len(x_cut_start)):
        if x_cut_start[i] - 3 > 0 and x_cut_end[i] + 3 < w - 1:
            element = img[0:h, x_cut_start[i] - 3:x_cut_end[i] + 3]  # cut the element from the image
        else:
            element = img[0:h, x_cut_start[i]:x_cut_end[i]]
        element_name = "el" + str(element_number).zfill(5) + ".jpg"  # generate the element name
        if 5 < len(element[0]) < 40:  # if the element is valid
            try:  # if the element is not null
                cv2.imwrite(path + element_name, element)  # save the elements in the directory
            except:  # else, skip that element
                pass
        elif len(element[0]) > 40:
            large_elements.append(element)
            large_elements_index.append(element_number)
        element_number = element_number + 1  # increase the indexing number

    index = 0
    for el in large_elements:
        el = crop_image(el)
        changed_el = erode_dilate(el)
        hist = find_histogram(changed_el)
        x_cut_start, x_cut_end = get_element_coordinates(changed_el, hist)

        h, w = el.shape[:2]  # image dimensions
        for i in range(len(x_cut_start)):
            if x_cut_start[i] - 3 > 0 and x_cut_end[i] + 3 < w - 1:
                element = el[0:h, x_cut_start[i] - 3:x_cut_end[i] + 3]  # cut the element from the image
            else:
                element = el[0:h, x_cut_start[i]:x_cut_end[i]]
            element_name = "el" + str(large_elements_index[index]).zfill(5) +\
                           "part" + str(i) + ".jpg"         # generate the element name
            if 4 < len(element[0]):
                try:  # if the element is not null
                    cv2.imwrite(path + element_name, element)  # save the elements in the directory
                except:  # else, skip that element
                    pass
        index = index + 1

    return element_number




path = ".\\resources\\img02\\"                                         # image location

images = []                                                            # get all the image names in the directory
for r, d, f in os.walk(path):                                          # r=root, d=directories, f = files
    for file in f:
        if '.jpg' in file:
            if 'el' not in file:
                images.append(file)

elementNumber = 0                                                     # indexing number for extracted elements
for image_name in images:                                             # process every slice
    img = cv2.imread(path + image_name)                               # read the image
    img = cv2.bitwise_not(img)                                        # convert colors
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                      # transform into gray img
    th, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # pixels have 1 of 2 values
    img = crop_image(thr)                                             # crop the left and right side of the image
    changed_img = erode_dilate(img)                                   # erode and dilate to filter out noise
    hist = find_histogram(changed_img)                                # find the histogram of symbol occurrences
    # find the start and end coordinates of the symbols
    x_cut_start, x_cut_end = get_element_coordinates(changed_img, hist)
    # get the updated element number and cut out all the symbols into separate images
    elementNumber = get_elements_from_image(path, x_cut_start, x_cut_end, img, elementNumber)

for fileName in os.listdir(path):                                   # delete redundant images from the previous step
    if fileName.startswith("slice"):
        os.remove(os.path.join(path, fileName))