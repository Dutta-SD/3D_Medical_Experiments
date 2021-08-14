import monai
import torch.nn as nn

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
