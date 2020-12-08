# local imports
from model.generator import PDELGenerator
from model.discriminator import PDELDiscriminator
from dataset.forced_isotropic_dataset import load_cutservice_file, ForcedIsotropicDataset

# x SPADE imports refactor out!
from SPADE.options.train_options import TrainOptions
from SPADE.models.networks.sync_batchnorm import DataParallelWithCallback


files = load_cutservice_file("dataset/prep/isotropic1024coarse_test32_16.h5")

epochs = range(1)


opt = TrainOptions().parse()
opt.normG = "spectralinstance"

netG = DataParallelWithCallback(PDELGenerator(opt),
                                device_ids=opt.gpu_ids)
netD = DataParallelWithCallback(PDELDiscriminator(),
                                device_ids=opt.gpu_ids)


for epoch in epochs:
    for i, data_i in enumerate([files]):
        x = netG.forward(data_i)
        dx = netD.forward(data_i, x)
        pass
