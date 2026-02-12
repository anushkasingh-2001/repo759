#!/usr/bin/env bash
set -euo pipefail

# Output CSVs
echo "n,ms" > tpb512.csv
echo "n,ms" > tpb16.csv

# n = 2^10 .. 2^29
for p in $(seq 10 29); do
  n=$((1<<p))

  # task3 prints time on line 1
  ms512=$(./task3 "$n" 512 | head -n 1)
  echo "$n,$ms512" >> tpb512.csv

  ms16=$(./task3 "$n" 16 | head -n 1)
  echo "$n,$ms16" >> tpb16.csv
done

echo "Wrote: tpb512.csv and tpb16.csv"
