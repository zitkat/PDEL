"""
Based on https://github.com/NVlabs/SPADE

Original copyright:

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys

import torch
from torch.utils.data import DataLoader

# local imports
from dataset.forced_isotropic_dataset import ForcedIsotropicDataset
from model.pdel_trainer import PDELTrainer
from util.iter_counter import IterationCounter
from util.options import parse_options
from util.visualizer import Visualizer


def main(argv):
    if argv is None:
        argv = sys.argv[1:]

    opt = parse_options(argv)
    opt.isTrain = True

    dataset = ForcedIsotropicDataset(root_dir=opt.dataset_path)
    split = [int(len(dataset) * s) for s in (.7, .1, .2)]
    data_train, _, _ = torch.utils.data.random_split(
            dataset, lengths=split, generator=torch.Generator().manual_seed(42))

    dataloader = DataLoader(data_train, batch_size=opt.batchSize, shuffle=True)

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
                iter_counter.record_current_errors(epoch,
                                                   iter_counter.epoch_iter,
                                                   losses,
                                                   iter_counter.time_per_iter)

            if iter_counter.needs_displaying():
                visualizer.save_paraview_snapshots(epoch,
                                                   iter_counter.epoch_iter,
                                                   time_i[0],
                                                   data_i[0],
                                                   trainer.get_latest_generated()
                                                   [0])

            if iter_counter.needs_saving():
                iter_counter.printlog('saving the latest model '
                                      f'(epoch {epoch}, '
                                      f'total_steps {iter_counter.total_steps_so_far})')
                trainer.save('latest')
                iter_counter.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            iter_counter.printlog('saving the model at the end of '
                                  'epoch {epoch}, '
                                  'iters {iter_counter.total_steps_so_far}')
            trainer.save('latest')
            trainer.save(epoch)


if __name__ == '__main__':
    main(None)
