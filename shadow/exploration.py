import cv2
import numpy as np
import cv2utils as u
import os
import config


def transform2(img, gamma):
    res = img.copy()
    B, G, R = cv2.split(res)
    R = u.gamma_corr(R, gamma=gamma)
    res = cv2.merge([B, G, R])
    return res


def transform3(img):
    res = u.bgr_to_hls(img.copy())
    B, G, R = cv2.split(res)
    return B


# GrayScale
if __name__ == "__main__":
    shadow_img_path = config.BASE_DIR / "ISTD/Img_shadow"

    for file in os.listdir(shadow_img_path):
        file_name = file.split(".")[0]
        img = cv2.imread(os.path.join(shadow_img_path, file))

        for gamma in [0.1, 0.2, 0.5, 0.75, 0.8, 1.1, 1.3, 1.4, 1.7, 2.0, 2.3, 3.1]:
            pth = os.path.join(config.BASE_DIR / "ISTD", "Img_temp", file_name)
            if not os.path.exists(pth):
                os.makedirs(pth)
            x = transform3(transform2(img, gamma))
            cv2.imwrite(os.path.join(pth, f"{gamma}__{file}"), x)
