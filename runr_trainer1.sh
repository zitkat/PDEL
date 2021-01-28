#!/bin/bash
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:ngpus=1:cl_gram=False:cl_doom=False:mem=10gb
#PBS -j oe
#PBS -o /storage/plzen1/home/zitkat/PDEL/
#PBS -m ae


# -- environment setup
#module add anaconda3-2019.10
#conda create -n PDELenv python=3.8
#source activate PDELenv
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
#conda install -c conda-forge h5py
#chmod u+x run_donwloader.sh

# -- actual run
module add anaconda3-2019.10
source activate PDELenv
cd /storage/plzen1/home/zitkat/PDEL/ || exit
export PYTHONPATH=/storage/plzen1/home/zitkat/PDEL/
today=$(date +%Y%m%d%H%M)
python train.py --name train1_nocitl -d /storage/plzen1/home/zitkat/datasets/JHTDB/isotropic128\
                -bs 1 --niter 100 &> ./trainlog"$today".txt
