#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -t 0-00:02:00
#SBATCH -J hw03_task1
#SBATCH -o task1-%j.out
#SBATCH -e task1-%j.err
#SBATCH --gres=gpu:1

./task1
