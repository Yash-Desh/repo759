#!/usr/bin/env zsh
#SBATCH –p instruction
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm-%j.out
#SBATCH -e FirstSlurm-%j.err
#SBATCH -c 2
#SBATCH -t 0-00:01:00

cd $SLURM_SUBMIT_DIR

hostname
