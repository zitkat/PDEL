

import argparse
import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader


# local imports
from model.networks.generator import PDELGenerator
from model.networks.discriminator import PDELDiscriminator
from dataset.forced_isotropic_dataset import ForcedIsotropicDataset

from model.sync_batchnorm import DataParallelWithCallback




def parse_gpu_ids(str_ids):
    # set gpu ids
    list_ids = str_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    # assert len(gpu_ids) == 0 or batchSize % len(gpu_ids) == 0, \
    #     "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
    #     % (batchSize, len(gpu_ids))

    return gpu_ids


def parse_options(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_path",
                        type=Path,
                        help="Path to root directory wtih h5 files")

    parser.add_argument('--gpu_ids', type=parse_gpu_ids, default=[0],
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='models are saved here')
    parser.add_argument('--model', type=str, default='fft',
                        help='which model to use')
    parser.add_argument('--norm_G', type=str, default='spectralinstance',
                        help='instance normalization or batch normalization')
    parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--norm_D', type=str, default='spectralinstance',
                        help='instance normalization or batch normalization')
    parser.add_argument('-bs', '--batch_size', type=int, default=1,
                        help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam')
    parser.add_argument('--niter', type=int, default=50,
                        help='# of iter at starting learning rate. This is NOT '
                             'the total #epochs. Total #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=0,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('-citl', '--constraint_in_the_loop', action="store_true",
                        # destination="constraint_in_the_loop",
                        help='Include constrain in the training loop.')
    parser.add_argument('--D_steps_per_G', type=int, default=1,
                        help='number of discriminator iterations per generator iterations.')

    opt, unknown = parser.parse_known_args(argv)

    return opt


def main(argv):
    if argv is None:
        argv = sys.argv[1:]

    opt = parse_options(argv)

    dataset = ForcedIsotropicDataset(root_dir=opt.dataset_path)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    netG = DataParallelWithCallback(PDELGenerator(opt),
                                    device_ids=opt.gpu_ids)
    optimG = optim.Adam(netG.parameters(), lr=opt.lr)

    netD = DataParallelWithCallback(PDELDiscriminator(),
                                    device_ids=opt.gpu_ids)

    optimD = optim.Adam(netD.parameters(), lr=opt.lr)

    epochs = range(1)

    for epoch in epochs:
        for i, (time_i, data_i) in enumerate(dataloader):
            if i % opt.D_steps_per_G == 0:
                optimG.zero_grad()
                x = netG.forward(data_i)
                dx = netD.forward(data_i, x)
                loss = torch.abs(1 - dx)
                loss.backward()
                optimG.step()

            optimD.zero_grad()
            dx = netD.forward(data_i, data_i)
            loss = torch.abs(1 - dx)
            loss.backward()
            optimD.step()


if __name__ == '__main__':
    main(None)
