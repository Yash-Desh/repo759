#!/usr/bin/env zsh

#SBATCH –p instruction

#SBATCH --job-name=FirstSlurm

#SBATCH -o FirstSlurm-%j.txt

#SBATCH -e FirstSlurm-%j.err

#SBATCH -c 2

#SBATCH -t 0-00:30:00

cd $SLURM_SUBMIT_DIR

hostname