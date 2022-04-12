import cv2
import numpy as np
import random


def getLevelRow(min_a, width):
    return [min_a for _ in range(width)]


# https://pastebin.com/knZvJRKL
def generate_shape(img_width, img_height, num_iters, padding):

    width = img_height
    height = img_width
    min_a, max_a = 0, 255

    drunk = {
        "wallCountdown": num_iters,
        "padding": padding,
        "x": int(width / 2),
        "y": int(height / 2),
    }

    level = [getLevelRow(min_a, width) for _ in range(height)]
    while drunk["wallCountdown"] >= 0:
        # print('hi')
        x = drunk["x"]
        y = drunk["y"]

        if level[y][x] == min_a:
            level[y][x] = max_a
            drunk["wallCountdown"] -= 1

        roll = random.randint(1, 4)

        if roll == 1 and x > drunk["padding"]:
            drunk["x"] -= 1

        if roll == 2 and x < width - 1 - drunk["padding"]:
            drunk["x"] += 1

        if roll == 3 and y > drunk["padding"]:
            drunk["y"] -= 1

        if roll == 4 and y < height - 1 - drunk["padding"]:
            drunk["y"] += 1

    level = np.array(level, dtype=np.uint8)
    return level


def generate_random_mask(
    img_height, img_width, num_iters, padding, full_path=None, save=False
):
    base_img = np.zeros(
        (img_width, img_height, 3),
        dtype=np.uint8,
    )

    # Black channel
    modified_alpha_channel = generate_shape(img_width, img_height, num_iters, padding)
    modified_alpha_channel = np.expand_dims(modified_alpha_channel, axis=-1)
    final_img = np.dstack((base_img, modified_alpha_channel))

   
    return final_img
