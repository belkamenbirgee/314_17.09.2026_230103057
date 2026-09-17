import threading
import time


THREADS = 4
INCREMENTS = 500_000


def worker(results, index):

    local_sum = 0

    for _ in range(INCREMENTS):
        local_sum += 1

    results[index] = local_sum


results = [0] * THREADS

threads = []

start = time.perf_counter()

for i in range(THREADS):

    t = threading.Thread(
        target=worker,
        args=(results, i)
    )

    threads.append(t)
    t.start()

for t in threads:
    t.join()


total = sum(results)

elapsed = time.perf_counter() - start


print("Expected:", THREADS * INCREMENTS)
print("Actual:", total)
print("Time:", elapsed)