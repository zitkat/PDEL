"""
Based on https://github.com/NVlabs/SPADE

Original copyright:

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from model.networks.sync_batchnorm.sync_batchnorm import DataParallelWithCallback
from model.pdel_model import PDELModel


class PDELTrainer:
    """
    Trainer creates the model and optimizers, and uses them to
    update the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pdel_model = PDELModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pdel_model = DataParallelWithCallback(self.pdel_model,
                                                       device_ids=opt.gpu_ids)
            self.pdel_model_on_one_gpu = self.pdel_model.module
        else:
            self.pdel_model_on_one_gpu = self.pdel_model

        self.generated = None
        self.optimizer_G, self.optimizer_D = \
            self.pdel_model_on_one_gpu.create_optimizers(opt)
        self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, l2loss, generated = self.pdel_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.g_losses.update({"L2": l2loss})
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pdel_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.pdel_model_on_one_gpu.save(epoch)

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
