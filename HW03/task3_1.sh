#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -J Task3

#SBATCH -o Task3-%j.out

#SBATCH -e Task3-%j.err

#SBATCH --cpus-per-task=20

#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
cd repo759/HW03
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

for ((counter=1; counter<=20; counter = counter+1))
do
./task3 1000000 $counter 32
printf "\n"
done
