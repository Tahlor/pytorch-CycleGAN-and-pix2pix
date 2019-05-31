#!/bin/bash

#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --output="runBW.slurm"
#SBATCH --constraint rhel7

echo $USER
if [ $USER=="tarch" ]; then
  email="taylor.archibald@byu.edu"
else
  email="masonfp@byu.edu"
fi

#SBATCH --mail-user=$email
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
#export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module

cat /etc/os-release
cat /etc/redhat-release

module purge
activate "/fslhome/tarch/anaconda3/envs/cycleGAN"
export PATH="/fslhome/$USER/anaconda3/envs/cycleGAN/bin:$PATH"

which python


cd "/fslhome/tarch/fsl_groups/fslg_hwr/compute/pytorch-CycleGAN"

# Has trainA and trainB folder in it
# TrainA = online
# TrainB = offline
data="/fslhome/tarch/compute/research/handwriting/MUNIT/datasets/handwriting"

# --results_dir
freq=500
python -u train.py --dataroot $data --name handwriting_cyclegan_BW --model cycle_gan --display_id 0 --no_flip --loadSizeY 64 --fineSizeY 64 --fineSizeX 1280 --loadSizeX 1280 --display_freq $freq --update_html_freq $freq --save_latest_freq $freq --input_nc 1 --output_nc 1

#--continue_train

# To run:
#sbatch ./run.sh
#sbatch /fslhome/tarch/compute/handwriting/run.sh 
#squeue -u tarch


#pip install git+https://github.com/Tahlor/utils.git
