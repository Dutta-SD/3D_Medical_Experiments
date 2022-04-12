# model definitions
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from medexp.models.fpn_unet import fpn_unet

unet_baseline = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
)

fpunet = fpn_unet.UNET3DwithAttention(
    in_channels=1,
    out_channels=2
)
