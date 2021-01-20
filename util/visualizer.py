"""
Based on https://github.com/NVlabs/SPADE

Original copyright:

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import time
from . import util
from dataset import ensured_path
from dataset.forced_isotropic_dataset import save_paraview_snapshot
from pathlib import Path

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x




class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = False  # opt.isTrain and opt.tf_log  # tensorboard logging
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if opt.isTrain:
            self.log_name = ensured_path(opt.checkpoints_dir / opt.name / 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

            self.snapshots_path = ensured_path(opt.checkpoints_dir / opt.name / "snapshots/",
                                               isdir=True)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    def save_paraview_snapshots(self, epoch, i, time_step, original, generated):
        ori_snap_path = self.snapshots_path / f"{epoch}_{i}_{time_step}_original"
        gen_snap_path = self.snapshots_path / f"{epoch}_{i}_{time_step}_generated"

        save_paraview_snapshot(ori_snap_path, original, time_step)
        save_paraview_snapshot(gen_snap_path, generated, time_step)





