#!/usr/bin/env zsh

#SBATCH –p instruction

#SBATCH --job-name=FirstSlurm

#SBATCH -o FirstSlurm-%j.txt

#SBATCH -e FirstSlurm-%j.err

#SBATCH -c 1

#SBATCH -t 0-00:30:00

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
g++ task6.cpp -Wall -O3 -std=c++17 -o task6
./task6 12