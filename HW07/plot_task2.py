import matplotlib.pyplot as plt

n_vals = []
time_vals = []

with open("task2_results.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        n_vals.append(int(parts[0]))
        time_vals.append(float(parts[1]))

plt.figure()
plt.loglog(n_vals, time_vals, marker='o', label='count()')
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.title("Count Scaling")
plt.legend()
plt.grid(True, which="both")
plt.savefig("task2.pdf", bbox_inches="tight")
plt.show()