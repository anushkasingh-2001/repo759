#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw09_t1
#SBATCH -o task1-%j.out
#SBATCH -e task1-%j.err
#SBATCH -c 10

g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

for t in {1..10}; do
  ./task1 5040000 $t
done

