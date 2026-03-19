#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:05:00
#SBATCH -J test_hw08_t2
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH -c 20

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
./task2 64 4