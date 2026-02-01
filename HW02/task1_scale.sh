#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J task1_scale
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 1
#SBATCH --mem=16G

set -euo pipefail

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

# Collect timing data
echo "n,time_ms" > task1_times.csv

for p in {10..30}; do
  n=$((1<<p))
  echo "Running n=$n" 


  # We only need line1.
  if out=$(./task1 $n); then
    t=$(echo "$out" | head -n 1)
    echo "$n,$t" >> task1_times.csv
  else
    echo "$n,NA" >> task1_times.csv
  fi
done

# Make task1.pdf (plot time vs n)
python3 - << 'PY'
import csv
import matplotlib.pyplot as plt

ns, ts = [], []
with open("task1_times.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["time_ms"] == "NA":
            continue
        ns.append(int(row["n"]))
        ts.append(float(row["time_ms"]))

plt.figure()
plt.plot(ns, ts, marker="o")
plt.xscale("log", base=2)
plt.xlabel("n (array length)")
plt.ylabel("Time taken by scan (ms)")
plt.title("Task 1 Scaling Analysis")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("task1.pdf")
print("Wrote task1.pdf")
PY
