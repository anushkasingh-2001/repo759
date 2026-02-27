#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw06_task1_bench
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

# Load CUDA (adjust if your Euler setup uses a different module)
module load nvidia/cuda/13.0.0

# Stop on errors
set -e

# Results file: columns = n  n_tests  avg_ms
RESULTS=task1_results.txt
echo "# n n_tests avg_ms" > $RESULTS

# Run n = 2^5 ... 2^11
for p in {5..11}; do
  n=$((2**p))

  # Choose repeats for stable timing (more repeats for small n, fewer for large n)
  if   [[ $n -le 32   ]]; then ntests=500
  elif [[ $n -le 64   ]]; then ntests=300
  elif [[ $n -le 128  ]]; then ntests=150
  elif [[ $n -le 256  ]]; then ntests=80
  elif [[ $n -le 512  ]]; then ntests=30
  elif [[ $n -le 1024 ]]; then ntests=10
  else ntests=4
  fi

  avg_ms=$(./task1 $n $ntests)
  echo "$n $ntests $avg_ms" | tee -a $RESULTS
done

# Generate task1.pdf plot using Python (matplotlib)
python3 - << 'PY'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xs, reps, ys = [], [], []
with open("task1_results.txt") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        n, nt, ms = line.split()
        xs.append(int(n))
        reps.append(int(nt))
        ys.append(float(ms))

plt.figure(figsize=(7,5))
plt.plot(xs, ys, marker='o')
plt.xlabel("n (matrix dimension)")
plt.ylabel("Average mmul time (ms)")
plt.title("HW06 Task1: cuBLAS mmul timing vs n")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("task1.pdf")
plt.savefig("task1.png", dpi=200)
PY

echo "Done. Generated:"
echo "  $RESULTS"
echo "  task1.pdf"
echo "  task1.png"