#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -J Task1

#SBATCH -o Task1-%j.out

#SBATCH -e Task1-%j.err

#SBATCH -c 1

#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
cd repo759/HW02
g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1
for ((counter=2**10; counter<=2**30; counter = counter*2))
do
./task1 $counter
printf "\n"
done
