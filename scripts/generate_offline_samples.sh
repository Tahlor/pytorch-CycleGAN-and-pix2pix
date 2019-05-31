#!/bin/bash

#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --output="runBWv3.slurm"
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


cd "/fslhome/$USER/fsl_groups/fslg_hwr/compute/pytorch-CycleGAN"

# Has trainA and trainB folder in it
# TrainA = online
# TrainB = offline

# --num_test
#python -u test.py --dataroot ./datasets/hwr --name handwriting_cyclegan_BW --model cycle_gan --input_nc 1 --output_nc 1 --loadSizeY 64 --fineSizeY 64 --fineSizeX 1280 --loadSizeX 1280 --results_dir "./results" --step 200
python -u test.py --dataroot ./datasets/hwr --name handwriting_cyclegan_BW_GT --model cycle_gan2 --input_nc 1 --output_nc 1 --loadSizeY 64 --fineSizeY 64 --fineSizeX 1280 --loadSizeX 1280 --results_dir "./results" --step 5800 --num_test 12149 --image_type fake_B

#--continue_train

# To run:
#sbatch ./run.sh
#sbatch /fslhome/tarch/compute/handwriting/run.sh 
#squeue -u tarch

#pip install git+https://github.com/Tahlor/utils.git