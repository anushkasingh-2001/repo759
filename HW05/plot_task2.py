import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

times = defaultdict(list)

with open(csv_file, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        N = int(row["N"])
        tpb = int(row["tpb"])
        t = float(row["time_ms"])
        times[(N, tpb)].append(t)

tpb_values = sorted(set(tpb for (_, tpb) in times.keys()))
N_values = sorted(set(N for (N, _) in times.keys()))

plt.figure(figsize=(8, 5))

for tpb in tpb_values:
    xs, ys = [], []
    for N in N_values:
        vals = times.get((N, tpb), [])
        if vals:
            xs.append(N)
            ys.append(sum(vals) / len(vals))  # average over repetitions
    plt.plot(xs, ys, marker='o', label=f"threads_per_block = {tpb}")

plt.xlabel("N")
plt.ylabel("Average time (ms) over repetitions")
plt.title("task2 reduction scaling")
plt.xscale("log", base=2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("task2.pdf")
print("Saved task2.pdf")