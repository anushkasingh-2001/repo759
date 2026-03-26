#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw09_t1_compare
#SBATCH -o task1_compare-%j.out
#SBATCH -e task1_compare-%j.err
#SBATCH -c 10

module purge
module load gcc/13.2.0

g++ task1.cpp cluster_false_shar.cpp -Wall -O3 -std=c++17 -o task1_false -fopenmp
g++ task1.cpp cluster.cpp            -Wall -O3 -std=c++17 -o task1_fixed -fopenmp

csv_file="task1_false_vs_fixed.csv"
echo "n,time_false_ms,time_fixed_ms" > "$csv_file"

t=10

for n in 630000 1260000 2520000 5040000 10080000 20160000; do
  total_false=0
  total_fixed=0

  for rep in {1..10}; do
    time_false=$(./task1_false $n $t | tail -n 1)
    time_fixed=$(./task1_fixed $n $t | tail -n 1)

    total_false=$(python3 - <<PY
total = float("$total_false")
time_ms = float("$time_false")
print(total + time_ms)
PY
)

    total_fixed=$(python3 - <<PY
total = float("$total_fixed")
time_ms = float("$time_fixed")
print(total + time_ms)
PY
)
  done

  avg_false=$(python3 - <<PY
total = float("$total_false")
print(total / 10.0)
PY
)

  avg_fixed=$(python3 - <<PY
total = float("$total_fixed")
print(total / 10.0)
PY
)

  echo "${n},${avg_false},${avg_fixed}" >> "$csv_file"
done

echo "Saved timings to $csv_file"
cat "$csv_file"