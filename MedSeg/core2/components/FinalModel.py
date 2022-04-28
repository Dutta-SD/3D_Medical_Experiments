# Final Model
import torch
import torch.nn as nn
from .FusionModule import Fusion
from .MultiAttentionHead import MultiAttentionHead
from .PatchEmbedding import PatchEmbedding
from .Projection import Projection
from .QKV import QKV
import config

# Model definition
class MedSegModel(nn.Module):
    def __init__(
        self,
        emb_dim=config.EMB_SIZE,
        patch_size=config.PATCH_SIZE,
        img_size=config.IMG_SIZE,
        n_proj_output_channels=config.PROJ_OP_CHANNEL,
    ):
        super(MedSegModel, self).__init__()

        # Common Reduction Stage
        self.reduction = nn.Sequential(
            Projection(n_output_channels=n_proj_output_channels, n_slices=155),
            PatchEmbedding(
                in_channels=n_proj_output_channels,
                patch_size=patch_size,
                emb_size=emb_dim,
                img_size=img_size,
            ),
        )

        # Separate QKV, Attention Stage
        self.qkv1 = QKV(emb_size=emb_dim)
        self.qkv2 = QKV(emb_size=emb_dim)
        self.attn1 = MultiAttentionHead(emb_size=emb_dim)
        self.attn2 = MultiAttentionHead(emb_size=emb_dim)

        # Single Fusion Module
        self.fusion_output = Fusion()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor)->torch.Tensor:
        out1 = self.reduction(x1)
        out2 = self.reduction(x2)
        q1, k1, v1 = self.qkv1(out1)
        q2, k2, v2 = self.qkv2(out2)
        o1 = self.attn1(q1, k2, v1)
        o2 = self.attn2(q2, k1, v2)
        out = self.fusion_output(o1, o2)
