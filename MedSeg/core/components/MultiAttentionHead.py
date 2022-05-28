import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import reduce, rearrange
from .componentLogger import get_logger

_LOG = get_logger()


class MultiAttentionHead(nn.Module):
    """
    Multi Head Attention Block. Computes Attention.
    Adapted from:
    """

    def __init__(
        self,
        emb_size: int = 768,
        num_heads: int = 12,
        att_drop_p: float = 0.0,
        projection_drop_p: float = 0.2,
    ) -> nn.Module:

        super(MultiAttentionHead, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(att_drop_p)
        self.projection = nn.Sequential(nn.Linear(emb_size, emb_size),
                                        nn.Dropout(projection_drop_p))
        self.scaling = (self.emb_size // num_heads)**-0.5

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        """
        dot product, Q V^T, here we don't transpose before, so this is why
        the sum is made on the last index of  K
        This is obtained from QKV module. Compatible
        """
        _LOG.debug(
            f"Component [{type(self).__name__}] input shape {queries.shape}")

        energy = torch.einsum("bhij, bhkj -> bhik", queries,
                              keys) * self.scaling

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy, dim=-1)
        att = self.att_drop(att)
        # dot product
        out = torch.einsum("bhij, bhjk -> bhik ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        _LOG.debug(
            f"Component [{type(self).__name__}] Ouptut Shape {out.shape}")
        return out
