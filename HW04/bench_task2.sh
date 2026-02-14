#!/bin/bash
#SBATCH --job-name=hw04_task2_bench
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bench_task2_%j.out

module load nvidia/cuda/13.0.0

R=128
TPB1=1024
TPB2=512   # valid because 512 >= 257

echo "n,tpb,ms" > task2.csv

for p in $(seq 10 29); do
  n=$((2**p))

  ms1=$(./task2 $n $R $TPB1 | tail -n 1)
  echo "$n,$TPB1,$ms1" >> task2.csv

  ms2=$(./task2 $n $R $TPB2 | tail -n 1)
  echo "$n,$TPB2,$ms2" >> task2.csv
done

echo "Wrote task2.csv"
