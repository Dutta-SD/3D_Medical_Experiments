import pytorch_lightning as pl

from conv_blocks import ConvNormDownSampling, ResidualAddAndUpsample


class FPN_BackBone_3D(pl.LightningModule):
    """
    FPN backbone for the data-module

    [b_s, 1, 96, 96 96] expected for current task
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=256,
    ):
        super().__init__()

        self._down_layer_1 = ConvNormDownSampling(in_channels=1, out_channels=128)
        self._down_layer_2 = ConvNormDownSampling(in_channels=128, out_channels=256)
        self._down_layer_3 = ConvNormDownSampling(in_channels=256, out_channels=512)
        self._down_layer_4 = ConvNormDownSampling(in_channels=512, out_channels=1024)

        self._res_layer_3 = ResidualAddAndUpsample(
            low_dim_tensor_channel=1024,
            high_dim_tensor_channel=512,
            out_channels=256,
        )
        self._res_layer_2 = ResidualAddAndUpsample(
            low_dim_tensor_channel=256,
            high_dim_tensor_channel=256,
            out_channels=128,
        )
        self._res_layer_1 = ResidualAddAndUpsample(
            low_dim_tensor_channel=128, high_dim_tensor_channel=128, out_channels=64
        )

    def forward(self, ip):
        d1 = self._down_layer_1(ip)
        d2 = self._down_layer_2(d1)
        d3 = self._down_layer_3(d2)
        d4 = self._down_layer_4(d3)

        u4 = self._res_layer_3(d4, d3)
        u3 = self._res_layer_2(u4, d2)
        u2 = self._res_layer_1(u3, d1)

        return [u4, u3, u2]
