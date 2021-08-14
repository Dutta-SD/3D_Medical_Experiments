import monai
import torch.nn as nn
import pytorch_lightning as pl

class ConvNormDownSampling(pl.LightningModule):
    '''
    Implements Convolution + Normalisation
    '''
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size = 2,
        stride = 2,
        ):

        super().__init__()

        self._conv_layer = nn.Conv3d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
        )

        self._norm_layer = monai.networks.blocks.ADN(
            in_channels = out_channels
        )

    def forward(self, x):
        return self._norm_layer(self._conv_layer(x))


class ConvNormUpSampling(pl.LightningModule):
    '''
    Implements Up-Convolution + Normalisation
    '''
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size = 2,
        stride = 2,
        ):

        super().__init__()

        self._conv_layer = nn.ConvTranspose3d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
        )

        self._norm_layer = monai.networks.blocks.ADN(
            in_channels = out_channels
        )

    def forward(self, x):
        return self._norm_layer(self._conv_layer(x))

class ResidualAddAndUpsample(pl.LightningModule):
    '''
    Makes residual and 1x1 convolution block
    low_dim_tensor - expected 5D Tensor
    high_dim_tensor = expected 5D Tensor
    '''
    def __init__(self, low_dim_tensor_channel, high_dim_tensor_channel, out_channels):
        super().__init__()
        self._one_x_one_conv = nn.Conv3d(
            in_channels = high_dim_tensor_channel,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1
        )
        self._up_conv = nn.ConvTranspose3d(
            in_channels = low_dim_tensor_channel,
            out_channels = out_channels,
            kernel_size = 2,
            stride = 2,
        )
    
    def forward(self, low_dim_tensor, high_dim_tensor):
        x = self._one_x_one_conv(high_dim_tensor)
        y = self._up_conv(low_dim_tensor)

        return (x+y)