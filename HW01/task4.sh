#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:05:00
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm-%j.out
#SBATCH -e FirstSlurm-%j.err
#SBATCH -c 1

hostname