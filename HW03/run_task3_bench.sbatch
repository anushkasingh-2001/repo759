#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -t 0-00:20:00
#SBATCH -J hw03_task3
#SBATCH -o task3bench-%j.out
#SBATCH -e task3bench-%j.err
#SBATCH --gres=gpu:1

module load nvidia/cuda/13.0.0

# Compile (safe to compile inside job)
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# Collect timings
bash bench_task3.sh

# Generate PDF plot
python3 plot_task3.py

ls -l tpb512.csv tpb16.csv task3.pdf
