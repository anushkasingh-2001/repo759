#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -t 0-00:02:00
#SBATCH -J hw03_task2
#SBATCH -o task2-%j.out
#SBATCH -e task2-%j.err
#SBATCH --gres=gpu:1

./task2
