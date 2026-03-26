import csv
import matplotlib.pyplot as plt

n_vals = []
time_vals = []

with open("task4_times.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        n_vals.append(int(row["n"]))
        time_vals.append(float(row["time_ms"]))

plt.figure(figsize=(6, 4))
plt.plot(n_vals, time_vals, marker="o")
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.title("Task 4: Convolution Time vs. n")
plt.grid(True)
plt.tight_layout()
plt.savefig("task4.pdf")
print("Saved plot to task4.pdf")