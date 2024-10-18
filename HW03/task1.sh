#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -J Task1

#SBATCH -o Task1-%j.out

#SBATCH -e Task1-%j.err

#SBATCH --cpus-per-task=20

#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
cd repo759/HW03
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

for ((counter=1; counter<=20; counter = counter+1))
do
./task1 1024 $counter
printf "\n"
done