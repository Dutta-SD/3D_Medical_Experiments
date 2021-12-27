import cv2
import numpy as np
import cv2utils as u

def transform1(img):
    res = u.bgr_to_gray(img)
    res = u.hist_eq(res)
    res = u.compress_by_2(res)
    res = u.dilate(res, kernel_size=(3,3), iter=2)
    res = u.kmeans_segment(res, 10, 20)
    res = u.expand_by_2(res)
    res = u.threshold_binary(res, inverse=True)

    return res

def transform2(img):
    res = img.copy()
    B, G, R = cv2.split(res)
    R = u.gamma_corr(R, gamma = 0.2)
    res = cv2.merge([B, G, R])
    return res

def transform3(img):
    res = u.bgr_to_hls(img.copy())
    B, G, R = cv2.split(res)
    return B

def view(img, x):
    cv2.imshow("Original", img)
    cv2.imshow("Shadow Mask", x)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# GrayScale
img = cv2.imread("./ISTD/Img_shadow/5-1.png")
x = transform3(transform2(img))
view(img, x)