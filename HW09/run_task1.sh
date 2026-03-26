#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw09_t1
#SBATCH -o task1-%j.out
#SBATCH -e task1-%j.err
#SBATCH -c 10

g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

csv_file="task1_times.csv"
echo "t,time_ms" > "$csv_file"

n=5040000

for t in {1..10}; do
  total=0

  # run 10 times and average
  for rep in {1..10}; do
    time_ms=$(./task1 $n $t | tail -n 1)
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

  echo "${t},${avg}" >> "$csv_file"
done

echo "Saved timings to $csv_file"
cat "$csv_file"