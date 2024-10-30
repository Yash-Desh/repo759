#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -J Task4

#SBATCH -o Task4-%j.out

#SBATCH -e Task4-%j.err

#SBATCH --cpus-per-task=8

#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
cd repo759/HW03
g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

export OMP_SCHEDULE="static,"

for ((counter=1; counter<=8; counter = counter+1))
do
./task3 300 10 $counter
printf "\n"
done
