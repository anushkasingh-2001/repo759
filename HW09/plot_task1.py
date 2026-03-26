import csv
import matplotlib.pyplot as plt

t_vals = []
time_false_vals = []
time_fixed_vals = []

with open("task1_false_vs_fixed.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t_vals.append(int(row["t"]))
        time_false_vals.append(float(row["time_false_ms"]))
        time_fixed_vals.append(float(row["time_fixed_ms"]))

plt.figure(figsize=(6, 4))
plt.plot(t_vals, time_false_vals, marker="o", label="False sharing")
plt.plot(t_vals, time_fixed_vals, marker="s", label="No false sharing")
plt.xlabel("t")
plt.ylabel("Time (ms)")
plt.title("Task 1: False Sharing vs No False Sharing")
plt.xticks(range(1, 11))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("task1_false_vs_fixed.pdf")
print("Saved plot to task1_false_vs_fixed.pdf")