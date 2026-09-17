
import threading
import random
import time

TOTAL_ITERATIONS = 100_000_000


def worker(iterations, results, index):
    local_hits = 0

    for _ in range(iterations):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            local_hits += 1

    results[index] = local_hits


def run_reduction(thread_count):

    threads = []
    results = [0] * thread_count

    base_iterations = TOTAL_ITERATIONS // thread_count
    remainder = TOTAL_ITERATIONS % thread_count

    start = time.perf_counter()

    for i in range(thread_count):

        iterations = base_iterations

        if i < remainder:
            iterations += 1

        thread = threading.Thread(
            target=worker,
            args=(iterations, results, i)
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_hits = sum(results)

    end = time.perf_counter()

    runtime = (end - start) * 1000
    pi = 4 * total_hits / TOTAL_ITERATIONS

    return runtime, pi


if __name__ == "__main__":

    thread_counts = [1, 2, 4, 8, 16, 32]

    baseline_time = None

    print("Threads | Runtime (ms) | Speedup | Efficiency | Pi")

    for thread_count in thread_counts:

        runtime, pi = run_reduction(thread_count)

        if baseline_time is None:
            baseline_time = runtime

        speedup = baseline_time / runtime
        efficiency = (speedup / thread_count) * 100

        print(
            f"{thread_count:7d} | "
            f"{runtime:12.2f} | "
            f"{speedup:7.2f}x | "
            f"{efficiency:10.2f}% | "
            f"{pi:.6f}"
        )