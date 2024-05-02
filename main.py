import cv2
import numpy as np
from matplotlib import pyplot as plt


def img_pre_show():
    image = cv2.imread("./data/000082.png")
    cv2.imshow("Original image", image)

    print()
    up_width = image.shape[0]*6
    up_height = image.shape[1]*6

    up_points = (up_width, up_height)
    resized_up = cv2.resize(image, up_points, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('Resized Up image by defining height and width', resized_up)

    return image, resized_up

img_ntr, img_resized = img_pre_show()

gray_image = cv2.cvtColor(img_ntr, cv2.COLOR_BGR2GRAY)
gray_image_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray_image', gray_image)
cv2.imshow('gray_image_resized', gray_image_resized)






median = cv2.medianBlur(img_resized, 5)
cv2.imshow('median', median)

Z = median.reshape((-1, 3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((median.shape))

gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('res2', res2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_resized, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', img_resized)


cv2.waitKey()
cv2.destroyAllWindows()
