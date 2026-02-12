import csv
import matplotlib.pyplot as plt

def read_csv(path):
    xs, ys = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["n"]))
            ys.append(float(row["ms"]))
    return xs, ys

x512, y512 = read_csv("tpb512.csv")
x16, y16 = read_csv("tpb16.csv")

plt.figure()
plt.plot(x512, y512, marker="o", label="512 threads/block")
plt.plot(x16, y16, marker="o", label="16 threads/block")

# since n is powers of 2
plt.xscale("log", base=2)

plt.xlabel("n")
plt.ylabel("Kernel time (ms)")
plt.title("vscale scaling on GPU")
plt.legend()
plt.tight_layout()

plt.savefig("task3.pdf")
print("Saved task3.pdf")
