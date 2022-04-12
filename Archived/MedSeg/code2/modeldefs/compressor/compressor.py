import torch
import torch.nn as nn
from torch import rand


class Compressor(nn.Module):
    """
    Compresses the given 5D tensor [(batch_dim, n_channels, depth, width, height)]
    into 2D tensor for using in Attention Modules. Uses 1x1 3D convolution.

    Takes in Single Batch of Tensors and Outputs a single 2D batch of Tensors
    Assumption: Each image tensor will have 1 channel and (D x W x H) size.
    """

    def __init__(self, n_output_channels):
        super(Compressor, self).__init__()
        self.n_output_channels = n_output_channels

    def forward(self, x):
        # We want 1x1 convolution
        # Remove channel Dimension
        x = x.squeeze(1)
        x = nn.Conv2d(x.shape[1], self.n_output_channels, 1)(x)
        return x.squeeze(1)


# Test
x = rand(8, 1, 200, 400, 600)
model = Compressor(1)
with torch.no_grad():
    op = model(x)
    print(op.shape)
