import csv
import matplotlib.pyplot as plt

t_vals = []
time_no_vals = []
time_simd_vals = []

with open("task2_times.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t_vals.append(int(row["t"]))
        time_no_vals.append(float(row["time_no_simd_ms"]))
        time_simd_vals.append(float(row["time_simd_ms"]))

plt.figure(figsize=(6, 4))
plt.plot(t_vals, time_no_vals, marker="o", label="Without simd")
plt.plot(t_vals, time_simd_vals, marker="s", label="With simd")
plt.xlabel("t")
plt.ylabel("Time (ms)")
plt.title("Task 2: Monte Carlo Time vs. t")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("task2.pdf")
print("Saved plot to task2.pdf")