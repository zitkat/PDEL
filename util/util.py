"""
Based on https://github.com/NVlabs/SPADE

Original copyright:

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import time
from pathlib import Path
import os
import argparse
from math import floor, ceil

import torch


def get_split(N, ratios):

    if sum(ratios) != 1.0:
        raise ValueError("Rations must sum to one.")

    rts = list(ratios)
    maxrt = max(rts)
    rts.remove(maxrt)

    split = [ceil(N * maxrt)] + [floor(N * rt) for rt in rts]
    return split




def copyconf(default_opt, **kwargs):
    """
    returns a configuration for creating a generator
    |default_opt| should be the opt of the current experiment
    |**kwargs|: if any configuration should be overriden, it can be specified here
    """
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def now():
    """
    :return: date and time as YYYYmmddhhMM
    """
    return time.strftime("%Y-%m-%d-%H-%M")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net


def ensured_path(path: Path, isdir=False):
    if isdir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path