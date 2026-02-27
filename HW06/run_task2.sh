#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw06_task2_bench
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

# Load CUDA (adjust if Euler uses a different module setup)
module load nvidia/cuda/13.0.0

set -e

# Results file: columns = n  last_scan_value  time_ms
RESULTS=task2_results.txt
echo "# n last_scan_value time_ms" > $RESULTS

# Run n = 2^10 ... 2^16 with threads_per_block = 1024 (per assignment)
TPB=1024

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

echo "Done. Results saved to $RESULTS"