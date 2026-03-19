#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw08_task2
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH -c 20

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

> hw8_task2_results.txt

for t in {1..20}
do
    time_ms=$(./task2 1024 $t | tail -n 1)
    echo "$t $time_ms" >> hw8_task2_results.txt
    echo "Done t=$t"
done