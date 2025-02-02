import torch
import torch.nn as nn


class ActorUNet(nn.Module):
    def __init__(self, n_input_channels=3, n_output_channels=3):
        super(ActorUNet, self).__init__()

        self.dconv_down1 = self._double_conv(n_input_channels, 64)
        self.dconv_down2 = self._double_conv(64, 128)
        self.dconv_down3 = self._double_conv(128, 256)
        self.dconv_down4 = self._double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up3 = self._double_conv(256 + 512, 256)
        self.dconv_up2 = self._double_conv(128 + 256, 128)
        self.dconv_up1 = self._double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_output_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
