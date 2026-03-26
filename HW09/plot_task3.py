import csv
import matplotlib.pyplot as plt

n_vals = []
time_vals = []

with open("task3_times.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        n_vals.append(int(row["n"]))
        time_vals.append(float(row["time_ms"]))

plt.figure(figsize=(6, 4))
plt.loglog(n_vals, time_vals, marker="o")
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.title("Task 3: Communication Time vs. n")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("task3.pdf")
print("Saved plot to task3.pdf")