import matplotlib.pyplot as plt

t_vals = []
time_vals = []

with open("hw8_task2_results.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        t_vals.append(int(parts[0]))
        time_vals.append(float(parts[1]))

plt.figure(figsize=(8, 6))
plt.plot(t_vals, time_vals, marker='o')
plt.xlabel("t")
plt.ylabel("Time (ms)")
plt.title("convolve time vs t")
plt.grid(True)
plt.tight_layout()
plt.savefig("hw8_task2.pdf")
plt.show()