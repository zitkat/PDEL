#!/bin/bash
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:ngpus=1:c:mem=10gb
#PBS -j oe
#PBS -o /storage/plzen1/home/zitkat/PDEL/
#PBS -m ae


# -- environment was set up as:
#module add anaconda3-2019.10
#conda create -n PDELenv python=3.8
#source activate PDELenv
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
#conda install -c conda-forge h5py
#chmod u+x run_donwloader.sh

# -- tested by:
##qsub -I -l select=1:ncpus=1:ngpus=1:mem=10gb -l walltime=1:00:00 -q gpu
##python train.py --name pdel_test_train -d /storage/plzen1/home/zitkat/datasets/JHTDB/isotropic128 \
##--niter 10 --debug

# -- actual run
module add anaconda3-2019.10
source activate PDELenv
cd /storage/plzen1/home/zitkat/PDEL/ || exit
today=$(date +%Y%m%d%H%M)
python train.py --name train1_citl -d /storage/plzen1/home/zitkat/datasets/JHTDB/isotropic128 \
                -bs 5 --niter 100 --citl &> ./trainlog"$today".txt
