#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw09_t2
#SBATCH -o task2-%j.out
#SBATCH -e task2-%j.err
#SBATCH -c 10

g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2_nosimd -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -DUSE_SIMD -o task2_simd -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

for t in {1..10}; do
    ./task2_nosimd 1000000 $t
done

for t in {1..10}; do
    ./task2_simd 1000000 $t
done