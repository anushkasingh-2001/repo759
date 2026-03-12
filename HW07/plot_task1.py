import matplotlib.pyplot as plt

n_vals = []
thrust_vals = []
cub_vals = []
hw05_vals = []

# Put your HW05 reduction timings here if you have them
# Example:
# hw05_map = {
#     1024: 0.020,
#     2048: 0.025,
# }
hw05_map = {}

with open("task1_results.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        n = int(parts[0])
        t_thrust = float(parts[1])
        t_cub = float(parts[2])

        n_vals.append(n)
        thrust_vals.append(t_thrust)
        cub_vals.append(t_cub)
        hw05_vals.append(hw05_map.get(n, None))

plt.figure()
plt.loglog(n_vals, thrust_vals, marker='o', label='Thrust')
plt.loglog(n_vals, cub_vals, marker='s', label='CUB')

valid_hw05_n = [n for n, v in zip(n_vals, hw05_vals) if v is not None]
valid_hw05_t = [v for v in hw05_vals if v is not None]
if valid_hw05_n:
    plt.loglog(valid_hw05_n, valid_hw05_t, marker='^', label='HW05 CUDA')

plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.title("Reduction Scaling")
plt.legend()
plt.grid(True, which="both")
plt.savefig("task1.pdf", bbox_inches="tight")
plt.show()