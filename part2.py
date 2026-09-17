
import threading
import random
import time

TOTAL_ITERATIONS = 50_000_000

synchronizedHits = 0
lock = threading.Lock()


def worker(iterations):
    global synchronizedHits

    for _ in range(iterations):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            with lock:
                synchronizedHits += 1


def run_multithreaded():
    global synchronizedHits
    synchronizedHits = 0

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

    pi = 4 * synchronizedHits / TOTAL_ITERATIONS
    runtime = (end - start) * 1000

    return pi, runtime


def run_single_thread():
    hits = 0

    start = time.perf_counter()

    for _ in range(TOTAL_ITERATIONS):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            hits += 1

    end = time.perf_counter()

    pi = 4 * hits / TOTAL_ITERATIONS
    runtime = (end - start) * 1000

    return pi, runtime


if __name__ == "__main__":

    print("===== SINGLE THREAD =====")

    pi, runtime = run_single_thread()

    print(f"Pi = {pi:.6f}")
    print(f"Runtime = {runtime:.2f} ms")

    print("\n===== 4 THREADS + LOCK =====")

    pi, runtime = run_multithreaded()

    print(f"Pi = {pi:.6f}")
    print(f"Runtime = {runtime:.2f} ms")