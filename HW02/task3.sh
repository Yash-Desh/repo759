#!/usr/bin/env zsh

#SBATCH –p instruction

#SBATCH -J Task3

#SBATCH -o Task3-%j.out

#SBATCH -e Task3-%j.err

#SBATCH -c 1

#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
cd repo759/HW02
g++ task3.cpp matmul.cpp -Wall -O3 -std=c++17 -o task3
./task3