#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw06_task1
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

# Load CUDA (adjust/remove if your cluster environment differs)
module load nvidia/cuda/13.0.0

# --- Compile task1 first ---
nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std=c++17 -o task1


# Example run:
# ./task1 <n> <n_tests>
# ./task1 1024 20

# Benchmark sizes 2^5 to 2^11 (prints: n avg_time_ms)
for p in {5..11}; do
  n=$((2**p))

  # choose a reasonable number of repeats for timing stability
  if   [[ $n -le 64   ]]; then ntests=200
  elif [[ $n -le 256  ]]; then ntests=100
  elif [[ $n -le 512  ]]; then ntests=40
  elif [[ $n -le 1024 ]]; then ntests=10
  else ntests=4
  fi

  t=$(./task1 $n $ntests)
  echo "$n $t"
done