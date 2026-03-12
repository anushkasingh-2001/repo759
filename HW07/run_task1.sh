#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw07_task1
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

module load nvidia/cuda/13.0

nvcc task1\ thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_thrust
nvcc task1\ cub.cu    -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_cub

for p in {10..20}
do
    n=$((2**p))
    thrust_time=$(./task1_thrust $n | tail -n 1)
    cub_time=$(./task1_cub $n | tail -n 1)
    echo "$n $thrust_time $cub_time"
done