"""
Script for downloading all needed dataset from
John Hopkins Turbulenece Database
"""

import os
from functools import wraps
from pathlib import Path
import numpy as np
import h5py

import pyJHTDB
from pyJHTDB import dbinfo as tdbinfo

from dataset import Shapes, ensured_path

data_folder = r"C:\Users\tozit\MLProjects\PDEL\PDEL\dataset"


# HOW TO OBTAIN AND LOAD DATA?
# 1. Loader requests new dataset frame from JHTDB for each epoch/evaluation
#   (possibly saves them)
#     a. request individual frames
#     b, request batches

# 2. Download all dataset, save them in
#     a. individual files
#     b. one file for input, one file for output
#     c. files for (input, ouput) * (train, validation, test)
#     d. files for (batch) * (input, ouput) * (train, validation, test)

#   ! ! ! !
# - v v v v  -
# I. Download individual files, save them as h5 or pickle
# II. a. Create torch.utils.dataset.Dataset for loading file by file
#     b. Create torch.utils.dataset.Dataset for loading chunks
#


# 128 x 128 x 128 x 3 ~~ 24 MB -> 120 GB
# 32 x 32 x 32 x 3 ~~ 0.526 KB = 526 kB -> 2.5 GB

def write_cut2hdf5(u, nt, file_path: Path):

    with h5py.File(file_path.with_suffix(".hdf5"), "w") as f:
        f.create_dataset("u", data=u)
        f.attrs.modify("nt", nt)


def read_cutfromhdf5(nt: int = None, file_path: Path = None):
    with h5py.File(file_path.with_suffix(".hdf5"), "w") as f:
        u = f["u"]

    return u


class DataDownLoader:

    def __init__(self, token_file="token"):
        with open(token_file, 'r') as f:
            auth_token = f.read().strip(" \n")

        self.lJHTDB = pyJHTDB.libJHTDB()
        self.lJHTDB.initialize()

        self.lJHTDB.add_token(auth_token)


    def get_signle_step(self, nt=16):
        u = self.lJHTDB.getbigCutout(
                t_start=nt, t_end=nt,
                start=np.array([1, 1, 1], dtype=np.int),
                end=np.array([1024, 1024, 1024], dtype=np.int),
                step=np.array([8, 8, 8], dtype=np.int))

        return u


    def get_multiple_steps(self):
        for t in range(16, 20):
            u = self.get_signle_step(t)
            file_path = Path(f"dltest/{t:04}_isotropic1024coarse_128")
            write_cut2hdf5(u, t, ensured_path(file_path))


    def finalize(self):
        self.lJHTDB.finalize()


if __name__ == '__main__':
    dl = DataDownLoader()
    dl.get_signle_step()


    dl.finalize()
