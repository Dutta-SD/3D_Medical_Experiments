import torch
import torch.nn as nn
from torch import rand
from einops.layers.torch import Rearrange, Reduce
from .componentLogger import get_logger
_LOG = get_logger()


class Projection(nn.Module):
    """
    Compresses the given 5D tensor [(batch_dim, n_channels, depth, width, height)]
    into 2D tensor for using in Attention Modules. Uses 1x1 3D convolution.

    Takes in Single Batch of Tensors and Outputs a single 2D batch of Tensors
    Assumption: Each image tensor will have 1 channel and (H, W, D) size.
    Thus, Each tensor has shape: (Batch_Size, Num_Channels, Height, Width, Slices/Depth)
    """

    def __init__(self, n_output_channels, n_slices):
        super(Projection, self).__init__()
        self.n_output_channels = n_output_channels
        self.n_slices = n_slices
        self.pipe = nn.Sequential(
            # Make it a 4D input, Treat depth as channel
            # Rearrange to make n_s first
            Rearrange("bs n_c h w n_s -> bs (n_s n_c) h w"),
            # 1x1 convolution to reduce number of channels, variable channel weights
            nn.Conv2d(self.n_slices, self.n_output_channels, 1),
        )

    def forward(self, x : torch.Tensor):
        _LOG.debug(f"Component [{type(self).__name__}] Input Shape {x.shape}")
        x = self.pipe(x)
        _LOG.debug(f"Component [{type(self).__name__}] Output Shape {x.shape}")
        return x


# # Test
# # Sigle Image -- (bs, n_c, h, w, n_s); n_c -- 1, bs -- 1
# x = rand(1, 1, 240, 240, 155)
# model = Projection(n_output_channels=3, n_slices=155)
# with torch.no_grad():
#     op = model(x)
#     print(op.shape)
