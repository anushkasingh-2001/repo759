#!/bin/bash
#SBATCH --job-name=hw04_task1_bench
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bench_task1_%j.out

module load nvidia/cuda/13.0.0

TPB1=1024
TPB2=256   # your "different threads per block"

echo "n,tpb,ms" > task1.csv

for p in $(seq 5 14); do
  n=$((2**p))

  # run once each; if you want, you can repeat and average (optional)
  ms1=$(./task1 $n $TPB1 | tail -n 1)
  echo "$n,$TPB1,$ms1" >> task1.csv

  ms2=$(./task1 $n $TPB2 | tail -n 1)
  echo "$n,$TPB2,$ms2" >> task1.csv
done

echo "Wrote task1.csv"
