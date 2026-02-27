#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J hw06_task2_memcheck
#SBATCH -o memcheck-%j.out
#SBATCH -e memcheck-%j.err
#SBATCH --gres=gpu:1 -c 1

module load nvidia/cuda/13.0.0

cuda-memcheck ./task2 1024 1024