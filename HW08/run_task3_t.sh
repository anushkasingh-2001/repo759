#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw08_task3_t
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH -c 20

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

> hw8_task3_t_results.txt

n=1000000
ts=BEST_TS

for t in {1..20}
do
    time_ms=$(./task3 $n $t $ts | tail -n 1)
    echo "$t $time_ms" >> hw8_task3_t_results.txt
    echo "Done t=$t"
done