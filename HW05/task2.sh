#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -J Task1

#SBATCH -o Task1-%j.out

#SBATCH -e Task1-%j.err

#SBATCH -c 1

#SBATCH --ntasks=1

#SBATCH --gpus-per-task=1

#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR


module load nvidia/cuda/11.8.0
module load gcc/11.3.0

nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2
./task2