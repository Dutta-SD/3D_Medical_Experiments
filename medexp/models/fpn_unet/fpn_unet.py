import torch.nn as nn
import torch
import pytorch_lightning as pl
import monai




# UNET
class UNET3DwithAttention(pl.LightningModule):
    '''
    UNet with FPN attention
    '''
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        # Layers
        self._fpn = FPN_BackBone_3D(in_channels=in_channels) # [256, 12, 12, 12][128, 24, 24, 24][64, 48, 48, 48]

        self._d_1 = ConvNormDownSampling(in_channels, 64) # 1x96x96x96 - 64x48x48x48
        self._d_2 = ConvNormDownSampling(64, 128) # 64x48x48x48 - 128x24x24x24
        self._d_3 = ConvNormDownSampling(128, 256) # 128x24x24x24 - 256x12x12x12
        self._d_4 = ConvNormDownSampling(256, 512) # 256x12x12x12 - 512x6x6x6

        self._u_4 = ConvNormUpSampling(512, 256) # 512x6x6x6 - 256x12x12x12
        self._u_3 = ConvNormUpSampling(256, 128) # 256x12x12x12 - 128x24x24x24
        self._u_2 = ConvNormUpSampling(128, 64) # 128x24x24x24 - 64x48x48x48
        self._u_1 = ConvNormUpSampling(64, out_channels) # 64x48x48x48 - 2x96x96x96

    def forward(self, x, noise):
        '''
        noise - random noise to generate attention maps
        x - input image

        noise is same dimensional as x
        '''
        # attention maps from FPN Backbone
        attn_mp = [torch.sigmoid(t) for t in self._fpn(noise)]

        # Downsample 
        o_d_1 = self._d_1(x)
        o_d_2 = self._d_2(o_d_1)
        o_d_3 = self._d_3(o_d_2)
        o_d_4 = self._d_4(o_d_3)

        x_low = o_d_4 # For understanding

        # Upsample and attention (dot product attention)
        o_u_4 = (self._u_4(x_low) + o_d_3) * attn_mp[0]
        o_u_3 = (self._u_3(o_u_4) + o_d_2) * attn_mp[1]
        o_u_2 = (self._u_2(o_u_3) + o_d_1) * attn_mp[2]
        o_u_1 = self._u_1(o_u_2)

        return o_u_1