#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --output="/fslhome/tarch/compute/handwriting/run.slurm"
#SBATCH --constraint rhel7

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module

cat /etc/os-release
cat /etc/redhat-release

module purge
source /fslhome/tarch/.bashrc
module load anaconda 
export PATH="/fslhome/tarch/anaconda3/bin:$PATH"
export PATH="/fslhome/tarch/anaconda3/envs/main/bin:$PATH"

which python

#/zhome/tarch/compute/handwriting
cd /fslhome/tarch/compute/research/handwriting2/pytorch-CycleGAN
data="/fslhome/tarch/compute/research/handwriting/MUNIT/datasets/handwriting"
python train.py --dataroot $data --name handwriting_cyclegan_BW --model cycle_gan --display_id 0 --no_flip --loadSizeY 64 --fineSizeY 64 --fineSizeX 1280 --loadSizeX 1280 --display_freq 400 --input_nc 1 --output_nc 1

#--continue_train

# To run:
#sbatch ./run.sh
#sbatch /fslhome/tarch/compute/handwriting/run.sh 
#squeue -u tarch


#pip install git+https://github.com/Tahlor/utils.git
