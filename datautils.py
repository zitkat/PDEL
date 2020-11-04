"""
Module producing data
"""

import numpy as nm
import h5py
import pathlib
import itertools
import torch




def load_file(file_path, ret_all=False):
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

# HOW TO OBTAIN AND LOAD DATA?
# 1. Loader requests new data frame from JHTDB for each epoch/evaluation
#   (possibly saves them)
#     a. request individual frames
#     b, request batches

# 2. Download all data, save them in
#     a. individual files
#     b. one file for input, one file for output
#     c. files for (input, ouput) * (train, validation, test)
#     d. files for (batch) * (input, ouput) * (train, validation, test)



# I. Download individual files, save them as h5 or pickle
# II. a. Create torch.utils.data.Dataset for loading file by file
#     b. Create torch.utils.data.Dataset for loading chunks
#


# 128 x 128 x 128 x 3 ~~ 24 MB -> 120 GB
# 32 x 32 x 32 x 3 ~~ 0.526 KB = 526 kB -> 2.5 GB






if __name__ == '__main__':
    load_file("data/prep/isotropic1024coarse_test32_16.h5")
    load_file("data/isotropic1024coarse_t100.h5")


