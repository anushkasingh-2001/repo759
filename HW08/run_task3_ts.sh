#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw08_task3_ts
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH -c 20

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

> hw8_task3_ts_results.txt

n=1000000
t=8

for p in {1..10}
do
    ts=$((2**p))
    time_ms=$(./task3 $n $t $ts | tail -n 1)
    echo "$ts $time_ms" >> hw8_task3_ts_results.txt
    echo "Done ts=$ts"
done