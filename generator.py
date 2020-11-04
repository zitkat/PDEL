from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from SPADE.models.networks.normalization import SPADE
from SPADE.models.networks.architecture import SPADEResnetBlock, \
    SPADEResnetBlock, spectral_norm
from SPADE.models.networks.base_network import BaseNetwork

from PDEL.architecture import SPADE3DResnetBlock


class PDELGenerator(BaseNetwork):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, opt):
        super().__init__()

        self.sin = (3,) + (32,) * 3  # shape input
        self.sinc = 3  # shape input channels
        self.sout = (3,) + (128,) * 3  # shape output
        self.soutc = 3  # shape output channels
        self.slatent = (3, 4, 4, 4)

        self.interpolate = F.interpolate
        self.conv1 = nn.Conv3d(self.sinc, 512, kernel_size=3, padding=1)
        self.sup1 = SPADE3DResnetBlock(512, 512, opt)  # 3D convolution
        self.sup2 = SPADE3DResnetBlock(512, 256, opt)
        self.sup3 = SPADE3DResnetBlock(256, 128, opt)
        self.sup4 = SPADE3DResnetBlock(128, 64, opt)
        self.sup5 = SPADE3DResnetBlock(64, 16, opt)

        self.up = nn.Upsample(scale_factor=2)

        self.conv2 = nn.Conv3d(16, self.soutc, kernel_size=3, padding=1)

    def forward(self, segmap):

        x = self.interpolate(segmap, size=self.slatent[1:])
        x = self.conv1(x)
        x = self.sup1(x, segmap)
        x = self.up(x)
        x = self.sup2(x, segmap)
        x = self.up(x)
        x = self.sup3(x, segmap)
        x = self.up(x)
        x = self.sup4(x, segmap)
        x = self.up(x)
        x = self.sup5(x, segmap)
        x = self.up(x)

        x = self.conv2(x)

        x = torch.fft(x, 1)

        # TODO constraints

        return x



