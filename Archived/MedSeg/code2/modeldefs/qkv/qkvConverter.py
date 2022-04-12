import torch
import torch.nn as nn
from torch import rand

class QKV(nn.Module):
    """
    Converts Given 2D image to QKV value.
    Step Before Attention
    """
    def __init__(self, seed = 0):
        super(QKV, self).__init__()
        self.seed = seed

    def forward(self, x):
        pass


