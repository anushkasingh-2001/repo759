#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw09_t3
#SBATCH -o task3-%j.out
#SBATCH -e task3-%j.err
#SBATCH --ntasks-per-node=2

module load mpi/mpich/4.0.2

mpicxx task3.cpp -Wall -O3 -o task3

csv_file="task3_times.csv"
echo "n,time_ms" > "$csv_file"

for p in {1..25}; do
  n=$((2**p))

  # If you want averaging to reduce noise, keep this block at 10 runs.
  total=0
  for rep in {1..10}; do
    time_ms=$(srun -n 2 ./task3 $n | tail -n 1)

    total=$(python3 - <<PY
total = float("$total")
time_ms = float("$time_ms")
print(total + time_ms)
PY
)
  done

  avg=$(python3 - <<PY
total = float("$total")
print(total / 10.0)
PY
)

  echo "${n},${avg}" >> "$csv_file"
done

echo "Saved timings to $csv_file"
cat "$csv_file"