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
            # Reduce("bs nc h w -> bs 1 h w", reduction="mean"),
            # Reduce("bs nc h w -> bs (m nc) h w", m=3, reduction="copy"),
            # 1x1 convolution to reduce number of channels, variable channel weights
            nn.Conv2d(self.n_slices, self.n_output_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _LOG.debug(f"Component [{__name__}] Input Shape {x.shape}")
        x = self.pipe(x)
        _LOG.debug(f"Component [{__name__}] Output Shape {x.shape}")
        return x
