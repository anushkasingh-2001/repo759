import csv
import matplotlib.pyplot as plt

data = {}
with open("task2.csv", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        n = int(row["n"])
        tpb = int(row["tpb"])
        ms = float(row["ms"])
        data.setdefault(tpb, []).append((n, ms))

plt.figure()

for tpb, pts in sorted(data.items()):
    pts.sort(key=lambda x: x[0])
    xs = [n for n, _ in pts]
    ys = [ms for _, ms in pts]
    plt.plot(xs, ys, marker="o", label=f"{tpb} threads/block")

plt.xscale("log", base=2)  # n = 2^p
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.title("Task 2: stencil scaling (R = 128)")
plt.legend()
plt.tight_layout()
plt.savefig("task2.pdf")
print("Saved task2.pdf")
