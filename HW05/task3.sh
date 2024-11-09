#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -J Task3

#SBATCH -o Task3-%j.out

#SBATCH -e Task3-%j.err

#SBATCH -c 1

#SBATCH --ntasks=1

#SBATCH --gpus-per-task=1

#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR


module load nvidia/cuda/11.8.0
module load gcc/9.4.0

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3


for ((counter=2**10; counter<=2**29; counter = counter*2))
do
./task3 counter
printf "\n"
done