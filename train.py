

import argparse
import sys

import torch

# local imports
from model.generator import PDELGenerator
from model.discriminator import PDELDiscriminator
from dataset.forced_isotropic_dataset import load_cutservice_file, ForcedIsotropicDataset

from model.sync_batchnorm import DataParallelWithCallback


files = load_cutservice_file("dataset/prep/isotropic1024coarse_test32_16.h5")



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
    parser.add_argument('--batchSize', type=int, default=1,
                        help='input batch size')
    parser.add_argument('--niter', type=int, default=50,
                        help='# of iter at starting learning rate. This is NOT '
                             'the total #epochs. Total #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=0,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('-citl', '--constraint_in_the_loop', action="store_true",
                        # destination="constraint_in_the_loop",
                        help='Include constrain in the training loop.')

    opt, unknown = parser.parse_known_args(argv)

    return opt


def main(argv):
    if argv is None:
        argv = sys.argv[1:]

    opt = parse_options(argv)

    netG = DataParallelWithCallback(PDELGenerator(opt),
                                    device_ids=opt.gpu_ids)
    netD = DataParallelWithCallback(PDELDiscriminator(),
                                    device_ids=opt.gpu_ids)
    epochs = range(1)

    for epoch in epochs:
        for i, data_i in enumerate([files]):
            x = netG.forward(data_i)
            dx = netD.forward(data_i, x)
            pass


if __name__ == '__main__':
    main(None)
