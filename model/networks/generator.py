import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model.networks.base_network import BaseNetwork

from model.networks.architecture import SPADE3DResnetBlock
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

        self.constrain = PDELDivergenceConstraint()
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

    def __init__(self):
        super().__init__()
        k_ = torch.arange(128)
        self.register_buffer("k", torch.empty((3, 128, 128, 128)))
        self.k[0], self.k[1], self.k[2] = torch.meshgrid(k_, k_, k_)
        self.register_buffer("kk", torch.sum(self.k ** 2, axis=0)[None, None, ...])


    def forward(self, f):
        F = torch.fft.fftn(f, dim=(2, 3, 4))
        dF = torch.einsum('kxyz,btxyz,txyz->btxyz',
                          -1j * self.k, F, -1j * self.k) / \
                   self.kk
        dF[torch.isnan(dF)] = 0
        hatF = F - dF
        hatf = torch.fft.ifftn(hatF, dim=(2, 3, 4))

        return hatf.real


if __name__ == '__main__':
    from dataset.forced_isotropic_dataset import load_cutservice_file
    file = load_cutservice_file(
        "../../dataset/prep/isotropic1024coarse_test128_16.h5")
    constr = PDELDivergenceConstraint() #.cuda(0)
    y = constr(file)

    pass
