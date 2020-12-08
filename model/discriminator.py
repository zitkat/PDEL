
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.base_network import BaseNetwork
from torch.nn.utils.spectral_norm import spectral_norm

from dataset import Shapes


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
        self.conv_last = nn.Conv3d(256, 1,
                                   kernel_size=4, stride=1, padding=padw)
        self.instance_norm = nn.InstanceNorm3d(256)



    def forward(self, lowres, highres):

        lowres_up = F.interpolate(lowres, size=Shapes.sout[1:])
        x = torch.cat((highres, lowres_up), dim=1)

        x = self.LReLu(self.conv1(x))
        x = self.LReLu(self.conv2(x))
        x = self.LReLu(self.conv3(x))
        x = self.LReLu(self.conv4(x))
        x = self.LReLu(self.instance_norm(x))
        x = self.conv_last(x)

        x = torch.log(x / (1 - x))


        return x




