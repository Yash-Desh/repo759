#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -J Task2

#SBATCH -o Task2-%j.out

#SBATCH -e Task2-%j.err

#SBATCH --cpus-per-task=20

#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
cd repo759/HW03
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

for ((counter=1; counter<=20; counter = counter+1))
do
./task2 1024 $counter
printf "\n"
done