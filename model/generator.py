from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from model.base_network import BaseNetwork

from model.architecture import SPADE3DResnetBlock
from dataset import Shapes


class PDELGenerator(BaseNetwork):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, opt):
        super().__init__()

        self.interpolate = F.interpolate
        self.conv1 = nn.Conv3d(Shapes.sinc, 512, kernel_size=3, padding=1)
        self.spade1 = SPADE3DResnetBlock(512, 512, opt)  # 3D convolution
        self.spade2 = SPADE3DResnetBlock(512, 256, opt)
        self.spade3 = SPADE3DResnetBlock(256, 128, opt)
        self.spade4 = SPADE3DResnetBlock(128, 64, opt)
        self.spade5 = SPADE3DResnetBlock(64, 16, opt)

        self.up = nn.Upsample(scale_factor=2)

        self.conv2 = nn.Conv3d(16, Shapes.soutc, kernel_size=3, padding=1)

    def forward(self, segmap):

        x = self.interpolate(segmap, size=Shapes.slatent[1:])
        x = self.conv1(x)
        x = self.spade1(x, segmap)
        x = self.up(x)
        x = self.spade2(x, segmap)
        x = self.up(x)
        x = self.spade3(x, segmap)
        x = self.up(x)
        x = self.spade4(x, segmap)
        x = self.up(x)
        x = self.spade5(x, segmap)
        x = self.up(x)

        x = self.conv2(x)

        # x = torch.fft(x, 1)

        # TODO constraints

        return x



