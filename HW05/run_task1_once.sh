#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J hw05p1a
#SBATCH -o hw05p1a-%j.out -e hw05p1a-%j.err
#SBATCH --gres=gpu:1 -c 1
#SBATCH --mem=4G

set -euo pipefail

# Compile
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Example run: ./task1 n block_dim
./task1 512 16