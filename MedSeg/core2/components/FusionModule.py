import torch
import torch.nn as nn
from .componentLogger import get_logger
from einops.layers.torch import Rearrange
import config

_LOG = get_logger()


class Fusion(nn.Module):

    def __init__(self):
        super(Fusion, self).__init__()
        self.k1 = 72
        self.k2 = 5
        self.k3 = 15
        # Upsampling Factor
        u1, u2, u3 = 1 / 2, (224 * 5 / 50), (224 * 5 / 720)
        self.layer1 = nn.Sequential(
            # nn.ConvTranspose2d(2,
            #                    1, (self.k1, self.k2),
            #                    stride=(self.k1, self.k2),
            #                    bias=False),
            # nn.Conv2d(1, config.N_SLICES, kernel_size=self.k3, stride=self.k3),
            Rearrange("b nx ny nz -> b 1 nx ny nz"),
            nn.Upsample(scale_factor=(u1, u2, u3)),
            nn.ReLU(),
            Rearrange("b nw nx ny nz -> b (nw nx) ny nz"),
            nn.Conv2d(1, config.N_SLICES, 1),
            nn.Conv2d(config.N_SLICES, config.N_SLICES, 5, stride=5),
            Rearrange("b nC h w -> b 1 h w nC"),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        _LOG.debug(f"Component [{type(self).__name__}] Input Shape {x1.shape}")
        # Add channel dimension
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        out = torch.cat([x1, x2], dim=1)
        _LOG.debug(
            f"Component [{type(self).__name__}] Shape of Final Tensor {out.shape}"
        )
        out = self.layer1(out)

        _LOG.debug(
            f"Component [{type(self).__name__}] Output Shape {out.shape}")
        return out
