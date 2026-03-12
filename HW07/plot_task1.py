import matplotlib.pyplot as plt

def read_two_col_file(filename):
    xs = []
    ys = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            xs.append(int(parts[0]))
            ys.append(float(parts[1]))
    return xs, ys

n_hw05, t_hw05 = read_two_col_file("hw05_task2_results.txt")
n_thrust, t_thrust = read_two_col_file("task1_thrust_results.txt")
n_cub, t_cub = read_two_col_file("task1_cub_results.txt")

plt.figure(figsize=(8, 6))

plt.loglog(n_hw05, t_hw05, marker='o', label='HW05 CUDA reduction')
plt.loglog(n_thrust, t_thrust, marker='s', label='HW07 Thrust reduction')
plt.loglog(n_cub, t_cub, marker='^', label='HW07 CUB reduction')

plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.title("Reduction time vs n")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("task1.pdf")
plt.show()