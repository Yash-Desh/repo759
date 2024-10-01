#!/usr/bin/env zsh

#SBATCH –p instruction

#SBATCH -J Task2

#SBATCH -o Task2-%j.out

#SBATCH -e Task2-%j.err

#SBATCH -c 1

#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
cd repo759/HW02
g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2
./task2 4 3