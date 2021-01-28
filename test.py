import sys

from math import ceil, floor
import numpy as np
import pandas as pd
from math import pi
import torch
import torch.fft
from torch.utils.data import DataLoader
from collections import namedtuple

# local imports
from dataset.forced_isotropic_dataset import ForcedIsotropicDataset
from model.networks.sync_batchnorm.sync_batchnorm import \
    DataParallelWithCallback
from model.pdel_model import PDELModel
from util.options import parse_options
from util.util import get_split
from util.visualizer import Visualizer

StatsRow = namedtuple("StatsRow", ["t",
                                   "dkenergy", "ddissipation", "dturnover",
                                   "gkenergy", "gdissipation", "gturnover",
                                   "l2"])


def compute_sol_stats(data):
    coors = ForcedIsotropicDataset.coors.to(data.device)
    dx = torch.mean(coors[1:] - coors[:-1])

    kenergy_tot = compute_kenergy(data)

    dissipation_tot = compute_dissipation(data, dx)

    turnover_t = compute_turnover(data, dx, kenergy_tot)

    return kenergy_tot, dissipation_tot, turnover_t


def compute_dissipation(data, dx):
    nu = 0.000185
    jacobu = torch.empty((3, 3, 127, 127, 127))
    jacobu[:, 0] = (data[:, 1:, 1:, 1:] - data[:, :-1, 1:, 1:]) / dx
    jacobu[:, 1] = (data[:, 1:, 1:, 1:] - data[:, 1:, :-1, 1:]) / dx
    jacobu[:, 2] = (data[:, 1:, 1:, 1:] - data[:, 1:, 1:, :-1]) / dx
    sigma = torch.empty((3, 3, 127, 127, 127))
    for i in range(3):
        for j in range(3):
            sigma[i, j] = (jacobu[i, j] + jacobu[j, i]) / 2
    dissipation = 2 * nu * torch.einsum("ij...,ij...->...", sigma, sigma)
    dissipation_tot = torch.mean(dissipation)
    return dissipation_tot


def compute_kenergy(data):
    kenergy = torch.einsum("d...,d...->...", data, data) / 2
    kenergy_tot = torch.mean(kenergy)
    return kenergy_tot


def compute_turnover(data, dx, kenergy_tot):
    F = torch.fft.fftn(data, dim=(1, 2, 3))
    mF = abs(F) / 128 ** 3
    mFF = mF ** 2
    k_end = int(128 / 2)
    rr = torch.arange(0, 128) - 128 / 2 + 1
    Rr = torch.roll(rr, int(128 / 2 + 1))
    X, Y, Z = torch.meshgrid(Rr, Rr, Rr)
    r = torch.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    k = torch.arange(1, k_end + 1) * dx
    spectrum = torch.zeros(k_end)
    bin_counter = torch.zeros(k_end)

    for n in range(1, k_end - 1):
        picker = (r * dx <= (k[n] + k[n]) / 2) & (
                    r * dx > (k[n] - k[n - 1]) / 2)
        spectrum[n] = torch.sum(mFF[:, picker])
        bin_counter[n] = torch.sum(picker)

    picker = (r * dx <= (k[1] + k[0]) / 2)
    spectrum[0] = torch.sum(mFF[:, picker])
    bin_counter[0] = torch.sum(picker)

    picker = (r * dx > (k[-1] + k[-2]) / 2) & (r * dx <= k[-1])
    spectrum[-1] = torch.sum(mFF[:, picker])
    bin_counter[-1] = torch.sum(picker)

    spectrum = spectrum / bin_counter
    up = torch.sqrt((2 / 3) * kenergy_tot)
    L = pi / (2 * up ** 2) * torch.sum(spectrum / k)
    turnover_t = L / up
    return turnover_t


def main(argv):
    if argv is None:
        argv = sys.argv[1:]

    opt = parse_options(argv)
    opt.isTrain = False

    dataset = ForcedIsotropicDataset(root_dir=opt.dataset_path)
    split = get_split(len(dataset), (.7, .1, .2))
    _, _, datatest = torch.utils.data.random_split(
            dataset, lengths=split, generator=torch.Generator().manual_seed(42))

    dataloader = DataLoader(datatest, batch_size=opt.batchSize, shuffle=True)

    # model = PDELModel(opt)
    # if len(opt.gpu_ids) > 0:
    #     model = DataParallelWithCallback(model, device_ids=opt.gpu_ids)
    # model.eval()

    visualizer = Visualizer(opt)
    data_stats = []
    generated_stats = []
    for i, (time_i, data_i) in enumerate(dataloader):

        # gloss, generated = model(data_i, mode='inference')
        for b in range(data_i.shape[0]):
            # print(f'Processing sample {i + b} time step {time_i[b]}')
            # visualizer.save_paraview_snapshots('test',
            #                                    i,
            #                                    time_i[b],
            #                                    data_i[b],
            #                                    generated[b]
            #                                    [0])
            naught = torch.zeros(1)
            dkenergy_tot, ddissipation_tot, dturnover_t = compute_sol_stats(data_i[b])
            gkenergy_tot, gdissipation_tot, gturnover_t = naught, naught, naught  # compute_sol_stats(generated[b])


            data_stats_row = StatsRow(*[tt.item() for tt in (time_i[b],
                                      dkenergy_tot, ddissipation_tot, dturnover_t,
                                      gkenergy_tot, gdissipation_tot, naught,
                                      naught)])
            data_stats.append(data_stats_row)

    stats_df = pd.DataFrame(data_stats)
    stats_df.to_csv(opt.checkpoints_dir / opt.name / "stats_df.csv")


if __name__ == '__main__':
    main(None)
