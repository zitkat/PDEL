"""
Script for downloading all needed dataset from
John Hopkins Turbulenece Database
"""

import os
import sys
import time
from functools import wraps
from pathlib import Path
import argparse
import csv

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


# 128 x 128 x 128 x 3 ~~ 24 MB x 5028 -> 120 GB
# 32 x 32 x 32 x 3 ~~ 0.526 MB = 526 kB -> 2.5 GB

def write_cut2hdf5(u, nt, file_path: Path):

    with h5py.File(file_path.with_suffix(".h5"), "w") as f:
        f.create_dataset("u", data=u)
        f.attrs.modify("nt", nt)


def read_cutfromhdf5(nt: int = None, file_path: Path = None):
    with h5py.File(file_path.with_suffix(".h5"), "w") as f:
        u = f["u"]

    return u


def now():
    """
    :return: date and time as YYYYmmddhhMM
    """
    return time.strftime("%Y%m%d%H%M")


class DataDownLoader:

    def __init__(self, token_file="token"):
        with open(token_file, 'r') as f:
            auth_token = f.read().strip(" \n")

        self.lJHTDB = pyJHTDB.libJHTDB()
        self.lJHTDB.initialize()

        self.lJHTDB.add_token(auth_token)


    def get_single_step(self, nt=16):
        u = self.lJHTDB.getbigCutout(
                t_start=nt, t_end=nt,
                start=np.array([1, 1, 1], dtype=np.int),
                end=np.array([1024, 1024, 1024], dtype=np.int),
                step=np.array([8, 8, 8], dtype=np.int))

        return u


    def save_steps(self, tstart: int, tend: int,  folder: Path):
        with open(ensured_path(folder / ("summary" + now() + ".csv")), "w", newline='') as summary_csv:
            summary_writer = csv.writer(summary_csv)
            summary_writer.writerow(["Timestep", "Downloadtime"])
            for t in range(tstart, tend + 1):
                wstart = time.time()
                u = self.get_single_step(t)
                wend = time.time()
                summary_writer.writerow([t, wend - wstart])
                summary_csv.flush()
                file_path = folder / Path(f"{t:04}_isotropic1024coarse_128")
                write_cut2hdf5(u, t, ensured_path(file_path))


    def finalize(self):
        self.lJHTDB.finalize()


def main(argv):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Script for detection of loupez in hives',
                                     epilog='(c) 2020 by T. Zitka and L. Picek, KKY UWB')

    parser.add_argument("-o", "--output",
                        help="Output directory, for everything: logfile, images and summary",
                        dest="output_dir",
                        metavar='path',
                        type=Path)
    parser.add_argument("-te", "--tend",
                        help="Last time to download, max 5028, min 0",
                        type=int,
                        default=16)
    parser.add_argument("-ts", "--tstart",
                        help="First time to download, max 5028, min 0",
                        type=int,
                        default=16)

    args = parser.parse_args(argv)

    dl = DataDownLoader()
    dl.save_steps(args.tstart, args.tend, args.output_dir)
    dl.finalize()


if __name__ == '__main__':
    main(None)


