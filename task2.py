import time
import threading
import statistics


ITERATIONS = 2_000_000


def run_test(stride):

    counters = [0] * (2 * stride)

    def worker(index):

        for _ in range(ITERATIONS):
            counters[index] += 1

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(stride,))

    start = time.perf_counter()

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    return time.perf_counter() - start


def median_test(stride):

    results = []

    for _ in range(3):

        t = run_test(stride)

        results.append(t)

        print(f"stride={stride}: {t:.4f}s")

    return statistics.median(results)


if __name__ == "__main__":

    print("Adjacent indices")
    adjacent = median_test(1)

    print("\nPadded indices")
    padded = median_test(16)

    print("\nResults:")
    print(f"Adjacent median: {adjacent:.4f}s")
    print(f"Padded median:   {padded:.4f}s")
    print(f"Slowdown:        {adjacent / padded:.2f}x")