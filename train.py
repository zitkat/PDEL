

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader


# local imports
from dataset.forced_isotropic_dataset import ForcedIsotropicDataset
from model.pdel_trainer import PDELTrainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer


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

    parser.add_argument('--name', type=str, default='pdel_train_test',
                        help='name of the experiment. It decides where to store '
                             'samples and models')
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
    parser.add_argument('-bs', '--batchSize', type=int, default=1,
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
    parser.add_argument('-citl', '--constraint_in_the_loop', action="store_true",
                        help='Include constrain in the training loop.')
    parser.add_argument('--D_steps_per_G', type=int, default=1,
                        help='number of discriminator iterations per generator '
                             'iterations.')
    parser.add_argument('--num_D', type=int, default=2,
                        help='number of discriminators to be used in multiscale')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue training: load the latest model')
    parser.add_argument('--gan_mode', type=str, default='hinge',
                        help='(ls|original|hinge)')

    parser.add_argument('--display_freq', type=int, default=100,
                        help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=5000,
                        help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=10,
                        help='frequency of saving checkpoints at the end of epochs')

    parser.add_argument('--debug', action='store_true',
                        help='only do one epoch and displays at each iteration')

    opt, unknown = parser.parse_known_args(argv)
    opt.isTrain = True
    return opt


def main(argv):
    if argv is None:
        argv = sys.argv[1:]

    opt = parse_options(argv)

    dataset = ForcedIsotropicDataset(root_dir=opt.dataset_path)

    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

    trainer = PDELTrainer(opt)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))

    # create tool for visualization
    visualizer = Visualizer(opt)

    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, (time_i, data_i) in enumerate(dataloader):
            iter_counter.record_one_iteration()

            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            trainer.run_discriminator_one_step(data_i)

            trainer.update_learning_rate(epoch)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses,
                                               iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                ...
                # TODO save some samples

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
            epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)


if __name__ == '__main__':
    main(None)
