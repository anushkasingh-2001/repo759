import csv
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

n_vals, t_int, t_float, t_double = [], [], [], []

with open(csv_file, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        n_vals.append(int(row["n"]))
        t_int.append(float(row["time_int_ms"]))
        t_float.append(float(row["time_float_ms"]))
        t_double.append(float(row["time_double_ms"]))

plt.figure(figsize=(8, 5))
plt.plot(n_vals, t_int, marker='o', label='int')
plt.plot(n_vals, t_float, marker='o', label='float')
plt.plot(n_vals, t_double, marker='o', label='double')
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.title("task1 runtime vs n")
plt.xscale("log", base=2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("task1.pdf")
print("Saved task1.pdf")