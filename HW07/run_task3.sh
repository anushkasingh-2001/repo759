#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J hw07_task3
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH -N 1 -c 4

g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
./task3