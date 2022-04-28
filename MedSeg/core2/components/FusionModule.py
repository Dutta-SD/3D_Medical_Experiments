import torch
import torch.nn as nn
from .componentLogger import get_logger
from einops.layers.torch import Rearrange

_LOG = get_logger()


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(2, 1, (72, 5), stride=(72, 5), bias=False),
            nn.Conv2d(1, 155, kernel_size=15, stride=15),
            Rearrange("b nC h w -> b 1 h w nC"),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        _LOG.debug(f"Component [{type(self).__name__}] Input Shape {x1.shape}")
        # Add channel dimension
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        out = torch.cat([x1, x2], dim=1)
        out = self.layer1(out)

        _LOG.debug(f"Component [{type(self).__name__}] Output Shape {out.shape}")
        return out
