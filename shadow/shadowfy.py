import cv2
import rand_gen as gen
import numpy as np


def shadowfy(img_array, alpha, num_iters, padding, k_shape):
    img = img_array

    height, width = img.shape[1], img.shape[0]
    beta = 1 - alpha

    mask = gen.generate_shape(width, height, num_iters, padding)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.bitwise_not(mask)
    mask = cv2.blur(mask, k_shape)
    # mask = cv2.dilate(mask, k_shape)

    masked_img = cv2.bitwise_and(img, mask)

    final_img = cv2.addWeighted(img, alpha, masked_img, beta, 0)

    return mask, final_img

if __name__ == "__main__":

    alpha = 0.75
    beta = 1 - alpha
    num_iters = 10000
    padding = 3
    k_shape = (5, 5)

    for i in range(1, 6):

        file_name = f"{i}-1.png"
        file_path = f"./ISTD/Img_shadow/{file_name}"
        save_path = f"./output/rand_shadow/transp-{file_name}"

        img = cv2.imread(file_path)
        mask, final_img = shadowfy(img, alpha, num_iters, padding, k_shape)

        cv2.imwrite(save_path, final_img)
