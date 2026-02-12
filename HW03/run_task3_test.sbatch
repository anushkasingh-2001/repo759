#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -t 0-00:05:00
#SBATCH -J hw03_task3_test
#SBATCH -o task3test-%j.out
#SBATCH -e task3test-%j.err
#SBATCH --gres=gpu:1

./task3 1024
