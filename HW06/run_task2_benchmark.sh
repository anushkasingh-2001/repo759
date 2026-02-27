#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw06_task2_bench
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

# Load CUDA (adjust/remove if your Euler setup differs)
module load nvidia/cuda/13.0.0

set -e

RESULTS=task2_results.txt
echo "# n last_scan_value time_ms" > $RESULTS

TPB=1024

# Run n = 2^10 ... 2^16
for p in {10..16}; do
  n=$((2**p))

  # task2 prints:
  # line 1 -> last scan value
  # line 2 -> elapsed time (ms)
  out=$(./task2 $n $TPB)
  last=$(echo "$out" | sed -n '1p')
  ms=$(echo "$out"   | sed -n '2p')

  echo "$n $last $ms" | tee -a $RESULTS
done

# Generate task2.pdf (time vs n)
python3 - << 'PY'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xs, ys = [], []
with open("task2_results.txt") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        n, last, ms = line.split()
        xs.append(int(n))
        ys.append(float(ms))

plt.figure(figsize=(7,5))
plt.plot(xs, ys, marker='o')
plt.xlabel("n")
plt.ylabel("Scan time (ms)")
plt.title("HW06 Task2: Scan timing vs n (threads_per_block=1024)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("task2.pdf")
plt.savefig("task2.png", dpi=200)
PY

echo "Done. Generated:"
echo "  $RESULTS"
echo "  task2.pdf"
echo "  task2.png"