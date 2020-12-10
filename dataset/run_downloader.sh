#!/bin/bash
#PBS -q iti
#PBS -l walltime=40:00:00
#PBS -l select=1:ncpus=1:ngpus=1:cl_gram=False:cl_doom=False:mem=10gb
#PBS -j oe
#PBS -o /storage/plzen1/home/grubiv/Scaffold_Analysis/UNet_alb_adam.log
#PBS -m ae


conda create --name DL_env python=3.9
conda install -y numpy h5py
pip install pyJHTDB

module add anaconda3-4.0.0
source activate DL_env
python /storage/plzen1/home/zitkat/PDEL/PDEL/dataset/data_downloader > /storage/plzen1/home/zitkat/PDEL/PDEL/dataset/log.txt
