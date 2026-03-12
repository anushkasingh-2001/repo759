#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw07_task2
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

module load nvidia/cuda/13.0

nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

> task2_results.txt

for p in {5..20}
do
    n=$((2**p))
    t=$(./task2 $n | tail -n 1)
    echo "$n $t" >> task2_results.txt
    echo "Done n=$n"
done