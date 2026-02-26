#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J hw05p2a
#SBATCH -o hw05p2a-%j.out -e hw05p2a-%j.err
#SBATCH --gres=gpu:1 -c 1
#SBATCH --mem=4G

set -euo pipefail

nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Example run: ./task2 N threads_per_block
./task2 1048576 1024