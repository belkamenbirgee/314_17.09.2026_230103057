
import threading
import random
import time

TOTAL_ITERATIONS = 50_000_000
totalHits = 0


def worker(iterations):
    global totalHits

    for _ in range(iterations):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            totalHits += 1


def run_part1():
    global totalHits
    totalHits = 0

    threads = []
    iterations_per_thread = TOTAL_ITERATIONS // 4

    start = time.perf_counter()

    for _ in range(4):
        thread = threading.Thread(
            target=worker,
            args=(iterations_per_thread,)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end = time.perf_counter()

    pi = 4 * totalHits / TOTAL_ITERATIONS
    runtime = (end - start) * 1000

    print(f"Pi = {pi:.6f}")
    print(f"Runtime = {runtime:.2f} ms")


if __name__ == "__main__":

    for i in range(5):
        print(f"\nRun {i + 1}:")
        run_part1()