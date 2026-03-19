#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:05:00
#SBATCH -J test_hw08_t1
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH -c 20

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp
./task1 64 4