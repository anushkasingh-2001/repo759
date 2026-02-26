#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-03:00:00
#SBATCH -J hw05p2b
#SBATCH -o hw05p2b-%j.out -e hw05p2b-%j.err
#SBATCH --gres=gpu:1 -c 1
#SBATCH --mem=4G

set -euo pipefail

nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

R=128
TPB1=1024        # required
TPB2=256         # your choice (can change)

CSV="task2_scaling_R${R}.csv"
echo "N,tpb,rep,time_ms" > "$CSV"

for p in {10..29}; do
    N=$((2**p))
    echo "===== N=${N} ====="

    for TPB in $TPB1 $TPB2; do
        echo "  TPB=${TPB}"

        for ((r=1; r<=R; r++)); do
            out=$(./task2 "$N" "$TPB")
            # task2 prints:
            # line1 = sum
            # line2 = time (ms)
            t=$(echo "$out" | sed -n '2p')
            echo "${N},${TPB},${r},${t}" >> "$CSV"
        done
    done
done

echo "Saved: $CSV"