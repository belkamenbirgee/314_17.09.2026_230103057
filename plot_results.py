import csv
import matplotlib.pyplot as plt

threads = []
empirical_speedup = []
theoretical_speedup = []

with open("results.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["Phase"] == "Phase3":
            threads.append(int(row["Threads"]))
            empirical_speedup.append(float(row["Speedup_or_Throughput"]))

        elif row["Phase"] == "Amdahl":
            theoretical_speedup.append(
                float(row["Speedup_or_Throughput"])
            )

plt.figure(figsize=(8, 5))

plt.plot(
    threads,
    empirical_speedup,
    marker="o",
    label="Empirical Speedup"
)

plt.plot(
    threads,
    theoretical_speedup,
    marker="o",
    label="Amdahl Theoretical"
)

plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Amdahl Reality Gap")
plt.xticks(threads)
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig("speedup_plot.png", dpi=300)

plt.show()