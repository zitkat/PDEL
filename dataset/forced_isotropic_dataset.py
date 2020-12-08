"""
Torch enabled dataset
"""

import numpy as nm
import h5py
import torch
import torch.utils.data as tdata
import random

from dataset import Shapes


class ForcedIsotropicDataset(tdata.Dataset):
    ...


def load_cutservice_file(file_path, ret_all=False):
    """
    Load hdf5 file from web cutout service
    """
    with h5py.File(file_path, mode='r', swmr=True) as f:
        keys = list(f)
        vel_keys = [(vel_key, int(vel_key.split("_")[-1]))
                    for vel_key in keys if "Velocity" in vel_key]
        vel_keys, times = zip(*vel_keys)
        coor_keys = [coor_key for coor_key in keys if "coor" in coor_key]

        data = torch.from_numpy(
                nm.stack([nm.moveaxis(f[vel_key], -1, 0) for vel_key in vel_keys]))
        coors = torch.from_numpy(
                nm.stack(
                nm.meshgrid(*[f[coor_key] for coor_key in coor_keys])))
        times = torch.from_numpy(nm.array(times))

    if ret_all:
        return data, coors, times

    return data


if __name__ == '__main__':
    load_cutservice_file("prep/isotropic1024coarse_test32_16.h5")
    load_cutservice_file("isotropic1024coarse_t100.h5")