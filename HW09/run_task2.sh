#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw09_t2
#SBATCH -o task2-%j.out
#SBATCH -e task2-%j.err
#SBATCH -c 10

g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2_nosimd -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -DUSE_SIMD -o task2_simd -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

csv_file="task2_times.csv"
echo "t,time_no_simd_ms,time_simd_ms" > "$csv_file"

n=1000000

for t in {1..10}; do
  total_no=0
  total_simd=0

  for rep in {1..10}; do
    time_no=$(./task2_nosimd $n $t | tail -n 1)
    time_simd=$(./task2_simd $n $t | tail -n 1)

    total_no=$(python3 - <<PY
total = float("$total_no")
time_ms = float("$time_no")
print(total + time_ms)
PY
)

    total_simd=$(python3 - <<PY
total = float("$total_simd")
time_ms = float("$time_simd")
print(total + time_ms)
PY
)
  done

  avg_no=$(python3 - <<PY
total = float("$total_no")
print(total / 10.0)
PY
)

  avg_simd=$(python3 - <<PY
total = float("$total_simd")
print(total / 10.0)
PY
)

  echo "${t},${avg_no},${avg_simd}" >> "$csv_file"
done

echo "Saved timings to $csv_file"
cat "$csv_file"