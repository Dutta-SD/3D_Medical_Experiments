# Generates QKV from Given 3D Medical Images
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor

"""
Source Code Has Been adapted from -- https://github.com/FrancescoSaverioZuppichini/glasses
LICENSE -- MIT
"""


class ViTTokens(nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor) -> List[Tensor]:
        b = x.shape[0]
        tokens = []
        for token in self.parameters():
            # for each token repeat itself over the batch dimension
            tokens.append(repeat(token, "() n e -> b n e", b=b))
        return tokens

    def __len__(self):
        return len(list(self.parameters()))


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
        tokens: nn.Module = ViTTokens,
    ):

        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.tokens = tokens(emb_size)
        self.positions = nn.Parameter(
            torch.randn((img_size // patch_size) ** 2 + len(self.tokens), emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        # get the tokens
        tokens = self.tokens(x)
        # prepend the tokens to the input
        x = torch.cat([*tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


# Test
x = torch.randn(1, 1, 240, 240)
print(PatchEmbedding(in_channels=1, img_size=240)(x).shape)
