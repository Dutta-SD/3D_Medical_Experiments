# generates QKV from Patch Embeddings
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from .componentLogger import get_logger

_LOG = get_logger()


class QKV(nn.Module):
    def __init__(
        self,
        emb_size: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = False,
    ):

        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3, bias=qkv_bias)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        _LOG.debug(f"Component [{type(self).__name__}] Input Shape to forward function is {x.shape}")
        qkv = rearrange(
            self.qkv(x),
            "b n (qkv h d) -> (qkv) b h n d",
            h=self.num_heads,
            qkv=3,
        )

        queries, keys, values = qkv
        _LOG.debug(f"Component [{type(self).__name__}] Shape of output query is: {queries.shape}")

        return queries, keys, values
