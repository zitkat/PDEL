"""
Based on https://github.com/NVlabs/SPADE

Original copyright:

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn

from model.networks.generator import PDELGenerator
from model.networks.discriminator import EnsembleDiscriminator
from model.networks.loss import GANLoss
from util import util


class PDELModel(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.netG, self.netD = self._initialize_networks(opt)

        if opt.isTrain:
            self.criterionGAN = GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)

    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        else:
            raise ValueError("|mode| is invalid")

    def _initialize_networks(self, opt):
        netG = PDELGenerator(opt)
        netD = EnsembleDiscriminator(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD
