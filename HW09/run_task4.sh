#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw09_t4
#SBATCH -o task4-%j.out
#SBATCH -e task4-%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 1

module load nvidia/cuda/13.0.0

clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
-Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_61 \
-march=native -o task4 convolve.cpp task4.cpp

csv_file="task4_times.csv"
echo "n,time_ms" > "$csv_file"

for n in 512 1024 2048 4096; do
  time_ms=$(./task4 $n)
  echo "${n},${time_ms}" >> "$csv_file"
done

echo "Saved timings to $csv_file"
cat "$csv_file"