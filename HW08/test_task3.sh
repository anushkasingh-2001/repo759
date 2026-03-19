#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:05:00
#SBATCH -J test_hw08_t3
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH -c 20

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
./task3 10000 4 32