"""
Based on https://github.com/NVlabs/SPADE

Original copyright:

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
from torch.nn.utils.spectral_norm import spectral_norm

from dataset import Shapes


class EnsembleDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = PDELDiscriminator()
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        return F.avg_pool3d(input, kernel_size=3,
                            stride=2, padding=[1, 1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        for name, D in self.named_children():
            out = D(input)
            result.append(out)
            input = self.downsample(input)

        return result


class PDELDiscriminator(BaseNetwork):

    def __init__(self):
        super().__init__()

        self.LReLu = nn.LeakyReLU(0.2, False)

        kw = 4
        stride = 2
        padw = int(np.ceil((kw - 1.0) / 2))
        self.conv1 = nn.Conv3d(
                2*Shapes.sinc, 16,
                kernel_size=4, stride=stride, padding=padw)
        self.conv2 = spectral_norm(nn.Conv3d(
                16, 64,
                kernel_size=4, stride=stride, padding=padw))
        self.conv3 = spectral_norm(nn.Conv3d(
                64, 128,
                kernel_size=4, stride=stride, padding=padw))
        self.conv4 = spectral_norm(nn.Conv3d(
                128, 256,
                kernel_size=4, stride=stride, padding=padw))
        self.conv_last = nn.Conv3d(
                256, 1,
                kernel_size=4, stride=1, padding=padw)
        self.instance_norm = nn.InstanceNorm3d(256)



    def forward(self, fake_and_real):

        fake_and_real = self.LReLu(self.conv1(fake_and_real))
        fake_and_real = self.LReLu(self.conv2(fake_and_real))
        fake_and_real = self.LReLu(self.conv3(fake_and_real))
        fake_and_real = self.LReLu(self.conv4(fake_and_real))
        fake_and_real = self.LReLu(self.instance_norm(fake_and_real))
        fake_and_real = self.conv_last(fake_and_real)


        return fake_and_real




