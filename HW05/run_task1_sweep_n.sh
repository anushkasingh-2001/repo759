#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-01:00:00
#SBATCH -J hw05p1b
#SBATCH -o hw05p1b-%j.out -e hw05p1b-%j.err
#SBATCH --gres=gpu:1 -c 1
#SBATCH --mem=4G

set -euo pipefail

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

BLOCK_DIM=16
CSV="task1_sweep_n_block${BLOCK_DIM}.csv"
echo "n,time_int_ms,time_float_ms,time_double_ms" > "$CSV"

for p in {5..14}; do
    n=$((2**p))
    echo "Running n=${n}, block_dim=${BLOCK_DIM}"

    out=$(./task1 "$n" "$BLOCK_DIM")

    # task1 prints 9 lines:
    # 1,2,3   -> int    (first, last, time)
    # 4,5,6   -> float  (first, last, time)
    # 7,8,9   -> double (first, last, time)
    t_int=$(echo "$out" | sed -n '3p')
    t_float=$(echo "$out" | sed -n '6p')
    t_double=$(echo "$out" | sed -n '9p')

    echo "${n},${t_int},${t_float},${t_double}" >> "$CSV"
done

echo "Saved: $CSV"