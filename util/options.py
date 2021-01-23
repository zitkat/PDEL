import argparse
from pathlib import Path

import torch


def parse_gpu_ids(str_ids):
    # set gpu ids
    list_ids = str_ids.split(',')
    gpu_ids = []
    for str_id in list_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    return gpu_ids


def parse_options(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='pdel_train_test',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument("-d", "--dataset_path",
                        type=Path,
                        help="Path to root directory with h5 files")
    parser.add_argument('--gpu_ids', type=parse_gpu_ids, default=[0],
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=Path, default='./checkpoints',
                        help='models are saved here')
    parser.add_argument('--model', type=str, default='fft',
                        help='which model to use')
    parser.add_argument('--norm_G', type=str, default='spectralinstance',
                        help='instance normalization or batch normalization')
    parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--norm_D', type=str, default='spectralinstance',
                        help='instance normalization or batch normalization')
    parser.add_argument('-bs', '--batchSize', type=int, default=1,  # original article 18
                        help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam')
    parser.add_argument('--niter', type=int, default=50,
                        help='# of iter at starting learning rate. This is NOT '
                             'the total #epochs. Total #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=0,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='momentum term of adam')
    parser.add_argument('--no_TTUR', action='store_true',
                        help='Use TTUR training scheme')
    parser.add_argument('-citl', '--constraint_in_the_loop',
                        action="store_true",
                        default=False,
                        help='Include constrain in the training loop.')
    parser.add_argument('--D_steps_per_G', type=int, default=1,
                        help='number of discriminator iterations per generator iterations.')
    parser.add_argument('--num_D', type=int, default=3,
                        help='number of discriminators to be used in multiscale')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue training: load the latest model')
    parser.add_argument('--which_epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')

    parser.add_argument('--gan_mode', type=str, default='hinge',
                        help='(ls|original|hinge)')

    parser.add_argument('--display_freq', type=int, default=1400,
                        help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=350,
                        help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=1400,
                        help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=10,
                        help='frequency of saving checkpoints at the end of epochs')

    parser.add_argument('--debug', action='store_true',
                        help='only do one epoch and displays at each iteration')

    opt, unknown = parser.parse_known_args(argv)

    assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
        "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
        % (opt.batchSize, len(opt.gpu_ids))

    return opt
