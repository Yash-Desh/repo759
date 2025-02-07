#!/usr/bin/env zsh

#SBATCH --partition=instruction
#SBATCH --time=00:010:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=Task1.out
#SBATCH --error=Task1.err
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
module load nvidia/cuda/11.8.0

# going into the subdirectory
# cd ece759/repo759/HW05

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

for ((i=5; i<15; i++)); do
	N=$((2 ** i))
	./task1 $N 32
done
