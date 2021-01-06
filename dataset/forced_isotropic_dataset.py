"""
Torch enabled dataset
"""

import numpy as nm
import h5py
import torch
import torch.utils.data as tdata
import random
import glob
from pathlib import Path

from dataset import Shapes


class ForcedIsotropicDataset(tdata.Dataset):
    """
    Simple class to work with data from forced isotropic 1024 turbulence dataset
    downloaded using data_downloader.

    Getting an item returns time step in simulation (which is not the same as
    index of an item) and the actual data of shape 3 x 128 x 128 x 128
    """

    def __init__(self, root_dir=Path("dataset/dltest/"), vel_key="u"):
        self.root_dir = Path(root_dir)
        self.vel_key = vel_key
        self.files = list(self.root_dir.glob("*.h5"))


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        file_path = self.files[idx]
        with h5py.File(file_path, mode='r', swmr=True) as f:
            data = torch.from_numpy(nm.moveaxis(f[self.vel_key], -1, 0))

        time = int(file_path.name.split("_")[0])

        return time, data


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
    # load_cutservice_file("prep/isotropic1024coarse_test32_16.h5")
    # load_cutservice_file("isotropic1024coarse_t100.h5")

    fid = ForcedIsotropicDataset()
    len(fid)

    r1 = fid[1]
    pass