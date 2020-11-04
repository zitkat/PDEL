

import sys
from collections import OrderedDict
from options.train_options import TrainOptions


# local imports
import datautils
from generator import PDELGenerator


files = datautils.load_file("data/prep/isotropic1024coarse_test32_16.h5")

epochs = range(10)


opt = TrainOptions().parse()
opt.normG = "spectralinstance"

for epoch in epochs:
    for i, data_i in enumerate([files]):
        netG = PDELGenerator(opt)
        x = netG.forward(data_i)
        pass
