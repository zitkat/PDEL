"""
Based on https://github.com/NVlabs/SPADE

Original copyright:

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from dataset import Shapes
from model.networks.generator import PDELGenerator
from model.networks.discriminator import EnsembleDiscriminator
from model.networks.loss import GANLoss
from util import util


class PDELModel(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.netG, self.netD = self._initialize_networks(opt)

        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.criterionGAN = GANLoss(
            opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def forward(self, data, mode):
        """
        Entry point for all calls involving forward pass
        of deep networks. We used this approach since DataParallel module
        can't parallelize custom functions, we branch to different
        routines based on |mode|.
        """
        lowres, highres = data[:, :, ::4, ::4, ::4], data


        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(lowres, highres)
            l2loss = torch.linalg.norm((generated - highres), dim=(1, 2, 3, 4))
            return g_loss, l2loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(lowres, highres)
            return d_loss
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    def _initialize_networks(self, opt):
        netG = PDELGenerator(opt)
        netD = EnsembleDiscriminator(opt)

        if opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
        return netG, netD

    def compute_generator_loss(self, lowres, highres):
        G_losses = {}

        fake_image = self.generate_fake(lowres)

        pred_fake, pred_real = self.discriminate(
            lowres, fake_image, highres)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        return G_losses, fake_image

    def compute_discriminator_loss(self, lowres, highres):
        D_losses = {}
        with torch.no_grad():
            generated = self.generate_fake(lowres)
            generated = generated.detach()
            generated.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            lowres, generated, highres)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def generate_fake(self, lowres):

        upsamled = self.netG(lowres)

        return upsamled

    def discriminate(self, lowres, generated, highres):
        """
        Given fake and real image, return the prediction of discriminator
        for each fake and real image.
        """
        lowres_up = F.interpolate(lowres, size=Shapes.sout[1:])
        fake_concat = torch.cat([lowres_up, generated], dim=1)
        real_concat = torch.cat([lowres_up, highres], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def divide_pred(self, pred):
        """
        Take the prediction of fake and real images from the combined batch
        """
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append(p[:p.size(0) // 2])
                real.append(p[p.size(0) // 2:])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
