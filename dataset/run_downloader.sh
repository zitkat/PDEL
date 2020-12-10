#!/bin/bash
#PBS -q iti
#PBS -l walltime=40:00:00
#PBS -l select=1:ncpus=1:ngpus=1:cl_gram=False:cl_doom=False:mem=10gb
#PBS -j oe
#PBS -o /storage/plzen1/home/zitkat/PDEL/
#PBS -m ae

#module add anaconda3-2019.10
#conda create --name DL_env python=3.9
#source activate DL_env
#conda install numpy h5py
## pyJHTDB requirements
#conda install sympy
#python -m pip install pyJHTDB

module add anaconda3-2019.10
source activate DL_env
cd /storage/plzen1/home/zitkat/PDEL/dataset/
export PYTHONPATH=/storage/plzen1/home/zitkat/PDEL/
today=$(date +%Y%m%d%H%M)
python data_downloader.py -ts 151 -te 151 -o . &> ./log"$today".txt
