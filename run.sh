#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1
##
#SBATCH --job-name=diagnosis
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
##
#SBATCH --gres=gpu:rtx3090:1

hostname
date

module add CUDA/11.3
module add ANACONDA/2020.11

/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python /home1/wonhyung64/Github/diagnosis/test.py 
