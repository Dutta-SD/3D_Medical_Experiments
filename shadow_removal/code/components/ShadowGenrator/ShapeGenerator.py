import numpy as np
import random


class PseudoShadowGenerator:
    """
    Generates Pseudo Shadow.
    This operates exclusively in the CPU. Do not move to GPU.
    """

    def __init__(self, n_channels=3, img_height=480, img_width=640):
        super(PseudoShadowGenerator, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.n_channels = n_channels
        # Numpy simulated Matrix
        self.simulated_matrix = np.zeros(
            (img_height, img_width),
            dtype=np.float32,
        )
        self.min_a = 0
        self.max_a = 255

    def simulate_shape(self, padding, num_iters) -> np.array:

        drunk = {
            "wallCountdown": num_iters,
            "padding": padding,
            "x": int(self.img_width / 2),
            "y": int(self.img_height / 2),
        }

        level = self.simulated_matrix

        while drunk["wallCountdown"] >= 0:
            x = drunk["x"]
            y = drunk["y"]

            if level[y][x] == self.min_a:
                level[y][x] = self.max_a
                drunk["wallCountdown"] -= 1

            roll = random.randint(1, 4)

            if roll == 1 and x > drunk["padding"]:
                drunk["x"] -= 1

            if roll == 2 and x < self.img_width - 1 - drunk["padding"]:
                drunk["x"] += 1

            if roll == 3 and y > drunk["padding"]:
                drunk["y"] -= 1

            if roll == 4 and y < self.img_height - 1 - drunk["padding"]:
                drunk["y"] += 1

        return level
