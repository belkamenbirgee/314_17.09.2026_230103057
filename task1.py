import time
from multiprocessing import Pool
import os


def cpu_work(n):
    count = 0

    for i in range(n):
        count += i * i

    return count


def run_test(workers, total_work):
    work_per_worker = total_work // workers

    start = time.perf_counter()

    with Pool(processes=workers) as pool:
        pool.map(cpu_work, [work_per_worker] * workers)

    end = time.perf_counter()

    return end - start


if __name__ == "__main__":

    total_work = 50_000_000

    print("CPU:", os.cpu_count())
    print()

    results = {}

    for workers in [1, 2, 4, 8, 12, 16]:

        t = run_test(workers, total_work)

        results[workers] = t

        print(f"Workers: {workers:2d} | Time: {t:.4f} s")

    print("\nSpeedup:")

    baseline = results[1]

    for workers, t in results.items():

        speedup = baseline / t
        efficiency = speedup / workers * 100

        print(
            f"{workers:2d} workers | "
            f"Speedup: {speedup:.2f}x | "
            f"Efficiency: {efficiency:.1f}%"
        )