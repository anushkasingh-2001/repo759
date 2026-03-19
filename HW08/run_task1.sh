#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw08_task1
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH -c 20

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

> hw8_task1_results.txt

for t in {1..20}
do
    time_ms=$(./task1 1024 $t | tail -n 1)
    echo "$t $time_ms" >> hw8_task1_results.txt
    echo "Done t=$t"
done