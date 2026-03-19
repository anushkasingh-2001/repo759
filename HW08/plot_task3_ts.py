import matplotlib.pyplot as plt

ts_vals = []
time_vals = []

with open("hw8_task3_ts_results.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        ts_vals.append(int(parts[0]))
        time_vals.append(float(parts[1]))

plt.figure(figsize=(8, 6))
plt.semilogx(ts_vals, time_vals, marker='o', base=2)
plt.xlabel("ts")
plt.ylabel("Time (ms)")
plt.title("msort time vs ts")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("hw8_task3_ts.pdf")
plt.show()