from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_network import BaseNetwork

from model.architecture import SPADE3DResnetBlock
from dataset import Shapes


class PDELGenerator(BaseNetwork):

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

        self.constrain = PDELDivergenceConstraint(opt)
        pass

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

        x = self.constrain(x)

        return x


class PDELDivergenceConstraint(nn.Module):

    def __init__(self, opt):
        super().__init__()
        device = opt.gpu_ids[0]
        k_ = torch.arange(128)
        self.register_buffer("k", torch.empty((3, 128, 128, 128)))
        self.k[0], self.k[1], self.k[2] = torch.meshgrid(k_, k_, k_)
        self.register_buffer("kk", torch.sum(self.k ** 2, axis=0))


    def forward(self, f):
        f_complex = torch.zeros(f.shape + (2,), device=f.device)
        f_complex[..., 0] = f
        F = f_complex.fft(3)
        # f_complex = torch.randn((1, 3, 128, 128, 128, 2))
        hatF = F - torch.einsum('txyz,bkxyzc,txyz->bkxyzc', self.k, F, self.k) / \
                   self.kk[None, None, ..., None]

        hatf = hatF.ifft(3)
        return hatf

