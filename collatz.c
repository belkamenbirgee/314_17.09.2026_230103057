/*
 * Parallel Computing Laboratory Practicum
 * The Amdahl Reality Gap
 *
 * Student ID: 230103057
 * Last 4 digits: 3057
 * N = 10,000,000 + (3057 * 1,000) = 13,057,000
 *
 * Compile:
 *     gcc -O2 -fopenmp collatz.c -o collatz
 *
 * Run:
 *     ./collatz
 *
 * Windows PowerShell:
 *     gcc -O2 -fopenmp collatz.c -o collatz.exe
 *     .\collatz.exe
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <inttypes.h>

#define N 13057000ULL
#define MOD 1000000007ULL

/* ---------------------------------------------------------
   Collatz stopping time
   --------------------------------------------------------- */
static inline uint32_t collatz_steps(uint64_t n)
{
    uint32_t steps = 0;

    while (n > 1) {
        if ((n & 1ULL) == 0)
            n >>= 1;
        else
            n = 3 * n + 1;

        steps++;
    }

    return steps;
}

/* ---------------------------------------------------------
   Sequential kernel
   Finds:
   - maximum stopping time
   - sum modulo MOD
   - number of values with >100 steps
   --------------------------------------------------------- */
static void sequential_kernel(
    uint64_t *checksum,
    uint32_t *max_steps,
    uint64_t *total_hits
)
{
    uint64_t sum = 0;
    uint32_t max = 0;
    uint64_t hits = 0;

    for (uint64_t i = 1; i <= N; i++) {

        uint32_t steps = collatz_steps(i);

        sum += steps;

        if (sum >= MOD)
            sum %= MOD;

        if (steps > max)
            max = steps;

        if (steps > 100)
            hits++;
    }

    *checksum = sum % MOD;
    *max_steps = max;
    *total_hits = hits;
}

/* ---------------------------------------------------------
   Sequential timing
   Run 1 = warmup
   Run 2 and Run 3 = measured
   --------------------------------------------------------- */
static double benchmark_sequential(
    uint64_t *checksum,
    uint32_t *max_steps,
    uint64_t *total_hits
)
{
    double start, end;
    double run1, run2, run3;

    uint64_t checksum_tmp;
    uint32_t max_tmp;
    uint64_t hits_tmp;

    /* Run 1 - cold / warmup */
    start = omp_get_wtime();

    sequential_kernel(
        &checksum_tmp,
        &max_tmp,
        &hits_tmp
    );

    end = omp_get_wtime();

    run1 = end - start;

    /* Run 2 */
    start = omp_get_wtime();

    sequential_kernel(
        &checksum_tmp,
        &max_tmp,
        &hits_tmp
    );

    end = omp_get_wtime();

    run2 = end - start;

    /* Save correct results */
    *checksum = checksum_tmp;
    *max_steps = max_tmp;
    *total_hits = hits_tmp;

    /* Run 3 */
    start = omp_get_wtime();

    sequential_kernel(
        &checksum_tmp,
        &max_tmp,
        &hits_tmp
    );

    end = omp_get_wtime();

    run3 = end - start;

    printf("\n========== PHASE 2: SEQUENTIAL ==========\n");
    printf("Run 1 (warmup): %.6f s\n", run1);
    printf("Run 2:          %.6f s\n", run2);
    printf("Run 3:          %.6f s\n", run3);

    printf("T_seq:          %.6f s\n",
           (run2 + run3) / 2.0);

    printf("Checksum:       %" PRIu64 "\n", *checksum);
    printf("Max steps:      %" PRIu32 "\n", *max_steps);
    printf("Hits >100:      %" PRIu64 "\n", *total_hits);

    return (run2 + run3) / 2.0;
}

/* ---------------------------------------------------------
   Parallel benchmark
   Uses OpenMP reduction for checksum, max and hits
   --------------------------------------------------------- */
static double parallel_kernel(
    int threads,
    uint64_t *checksum,
    uint32_t *max_steps,
    uint64_t *total_hits
)
{
    uint64_t sum = 0;
    uint32_t max = 0;
    uint64_t hits = 0;

    double start, end;

    omp_set_num_threads(threads);

    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum,hits) reduction(max:max) schedule(static)
    for (uint64_t i = 1; i <= N; i++) {

        uint32_t steps = collatz_steps(i);

        sum += steps;

        if (steps > max)
            max = steps;

        if (steps > 100)
            hits++;
    }

    end = omp_get_wtime();

    *checksum = sum % MOD;
    *max_steps = max;
    *total_hits = hits;

    return end - start;
}

/* ---------------------------------------------------------
   Phase 3
   Thread scaling
   --------------------------------------------------------- */
static void run_phase3(
    FILE *csv,
    double T_seq,
    uint64_t expected_checksum
)
{
    int thread_counts[] = {1, 2, 4, 8};
    int count = sizeof(thread_counts) / sizeof(thread_counts[0]);

    double times[4];

    printf("\n========== PHASE 3: OPENMP SCALING ==========\n");

    fprintf(csv,
            "PHASE3,threads,run1,run2,run3,avg_time,"
            "emp_speedup,theoretical_speedup,reality_gap,"
            "checksum,max_steps,hits\n");

    /*
     * First get Run 1/2/3 for every thread count.
     */
    for (int x = 0; x < count; x++) {

        int threads = thread_counts[x];

        double run1, run2, run3;

        uint64_t checksum;
        uint32_t max_steps;
        uint64_t hits;

        /* Run 1 - warmup */
        run1 = parallel_kernel(
            threads,
            &checksum,
            &max_steps,
            &hits
        );

        /* Run 2 */
        run2 = parallel_kernel(
            threads,
            &checksum,
            &max_steps,
            &hits
        );

        /* Run 3 */
        run3 = parallel_kernel(
            threads,
            &checksum,
            &max_steps,
            &hits
        );

        double avg = (run2 + run3) / 2.0;

        times[x] = avg;

        double empirical_speedup = T_seq / avg;

        /*
         * p will be calculated after k=2.
         * For now p = 0.
         */
        printf("\nThreads: %d\n", threads);
        printf("Run 1: %.6f s\n", run1);
        printf("Run 2: %.6f s\n", run2);
        printf("Run 3: %.6f s\n", run3);
        printf("Average: %.6f s\n", avg);
        printf("Empirical speedup: %.6f\n", empirical_speedup);

        if (checksum != expected_checksum) {
            printf("WARNING: checksum mismatch!\n");
        }

        fprintf(csv,
                "PHASE3,%d,%.9f,%.9f,%.9f,%.9f,%.9f,NA,NA,"
                "%" PRIu64 ",%" PRIu32 ",%" PRIu64 "\n",
                threads,
                run1,
                run2,
                run3,
                avg,
                empirical_speedup,
                checksum,
                max_steps,
                hits);
    }

    /*
     * Calculate p from measured k=2.
     *
     * S(2) = T_seq / T_2
     *
     * p = 2 * (1 - 1/S(2))
     */
    double S2 = T_seq / times[1];

    double p = 2.0 * (1.0 - (1.0 / S2));

    /*
     * Clamp p to valid Amdahl range.
     */
    if (p < 0.0)
        p = 0.0;

    if (p > 1.0)
        p = 1.0;

    printf("\n========== AMDahl FIT ==========\n");
    printf("S_emp(2): %.9f\n", S2);
    printf("Derived p: %.9f\n", p);
    printf("Sequential fraction (1-p): %.9f\n",
           1.0 - p);

    printf("\nThreads | S_emp | S_theo | Delta\n");
    printf("----------------------------------\n");

    for (int x = 0; x < count; x++) {

        int k = thread_counts[x];

        double S_emp = T_seq / times[x];

        double S_theo =
            1.0 / ((1.0 - p) + (p / (double)k));

        double delta = S_theo - S_emp;

        printf("%7d | %.4f | %.4f | %.4f\n",
               k,
               S_emp,
               S_theo,
               delta);
    }

    /*
     * Append theoretical results to CSV.
     */
    fprintf(csv, "\nAMDahl_fit,p,%.9f\n", p);

    fprintf(csv,
            "AMDahl_result,threads,S_emp,S_theo,Delta\n");

    for (int x = 0; x < count; x++) {

        int k = thread_counts[x];

        double S_emp = T_seq / times[x];

        double S_theo =
            1.0 / ((1.0 - p) + (p / (double)k));

        double delta = S_theo - S_emp;

        fprintf(csv,
                "AMDahl_result,%d,%.9f,%.9f,%.9f\n",
                k,
                S_emp,
                S_theo,
                delta);
    }
}

/* ---------------------------------------------------------
   Phase 4A - Naive false sharing
   --------------------------------------------------------- */
static double false_sharing_naive(
    int threads,
    uint64_t *total_hits
)
{
    int *hit_count =
        (int *)calloc(threads, sizeof(int));

    if (hit_count == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 1; i <= N; i++) {

        int tid = omp_get_thread_num();

        uint32_t steps = collatz_steps(i);

        if (steps > 100)
            hit_count[tid]++;
    }

    double end = omp_get_wtime();

    uint64_t total = 0;

    for (int i = 0; i < threads; i++)
        total += hit_count[i];

    *total_hits = total;

    free(hit_count);

    return end - start;
}

/* ---------------------------------------------------------
   Phase 4A - Reduction
   --------------------------------------------------------- */
static double false_sharing_reduction(
    int threads,
    uint64_t *total_hits
)
{
    uint64_t hits = 0;

    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:hits) schedule(static)
    for (uint64_t i = 1; i <= N; i++) {

        uint32_t steps = collatz_steps(i);

        if (steps > 100)
            hits++;
    }

    double end = omp_get_wtime();

    *total_hits = hits;

    return end - start;
}

/* ---------------------------------------------------------
   Phase 4A
   --------------------------------------------------------- */
static void run_phase4A(FILE *csv, int max_threads)
{
    printf("\n========== PHASE 4A: FALSE SHARING ==========\n");

    fprintf(csv,
            "\nPHASE4A,variant,threads,run1,run2,run3,"
            "avg_time,throughput,speedup_penalty_ratio,hits\n");

    uint64_t hits1, hits2;

    double r1_1, r1_2, r1_3;
    double r2_1, r2_2, r2_3;

    /*
     * Naive
     */
    r1_1 = false_sharing_naive(max_threads, &hits1);
    r1_2 = false_sharing_naive(max_threads, &hits1);
    r1_3 = false_sharing_naive(max_threads, &hits1);

    double avg_naive = (r1_2 + r1_3) / 2.0;

    /*
     * Reduction
     */
    r2_1 = false_sharing_reduction(max_threads, &hits2);
    r2_2 = false_sharing_reduction(max_threads, &hits2);
    r2_3 = false_sharing_reduction(max_threads, &hits2);

    double avg_reduction = (r2_2 + r2_3) / 2.0;

    double throughput_naive =
        (double)N / avg_naive;

    double throughput_reduction =
        (double)N / avg_reduction;

    /*
     * >1 means naive is slower than reduction.
     */
    double penalty_ratio =
        avg_naive / avg_reduction;

    printf("\nNaive false sharing:\n");
    printf("Run 1: %.6f s\n", r1_1);
    printf("Run 2: %.6f s\n", r1_2);
    printf("Run 3: %.6f s\n", r1_3);
    printf("Average: %.6f s\n", avg_naive);
    printf("Throughput: %.2f iter/sec\n",
           throughput_naive);
    printf("Hits: %" PRIu64 "\n", hits1);

    printf("\nReduction:\n");
    printf("Run 1: %.6f s\n", r2_1);
    printf("Run 2: %.6f s\n", r2_2);
    printf("Run 3: %.6f s\n", r2_3);
    printf("Average: %.6f s\n", avg_reduction);
    printf("Throughput: %.2f iter/sec\n",
           throughput_reduction);
    printf("Hits: %" PRIu64 "\n", hits2);

    printf("\nPenalty ratio (naive/reduction): %.4f\n",
           penalty_ratio);

    fprintf(csv,
            "PHASE4A,naive_false_sharing,%d,"
            "%.9f,%.9f,%.9f,%.9f,%.3f,%.6f,%"
            PRIu64 "\n",
            max_threads,
            r1_1,
            r1_2,
            r1_3,
            avg_naive,
            throughput_naive,
            penalty_ratio,
            hits1);

    fprintf(csv,
            "PHASE4A,reduction,%d,"
            "%.9f,%.9f,%.9f,%.9f,%.3f,1.000000,%"
            PRIu64 "\n",
            max_threads,
            r2_1,
            r2_2,
            r2_3,
            avg_reduction,
            throughput_reduction,
            hits2);
}

/* ---------------------------------------------------------
   Scheduling benchmark
   --------------------------------------------------------- */
static double schedule_test(
    int threads,
    int schedule_type,
    int chunk,
    uint64_t *checksum,
    uint32_t *max_steps
)
{
    uint64_t sum = 0;
    uint32_t max = 0;

    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    if (schedule_type == 0) {

        #pragma omp parallel for schedule(static) reduction(+:sum) reduction(max:max)
        for (uint64_t i = 1; i <= N; i++) {
            uint32_t steps = collatz_steps(i);
            sum += steps;

            if (steps > max)
                max = steps;
        }

    } else if (schedule_type == 1) {

        #pragma omp parallel for schedule(static,1000) reduction(+:sum) reduction(max:max)
        for (uint64_t i = 1; i <= N; i++) {
            uint32_t steps = collatz_steps(i);
            sum += steps;

            if (steps > max)
                max = steps;
        }

    } else if (schedule_type == 2) {

        #pragma omp parallel for schedule(dynamic,100) reduction(+:sum) reduction(max:max)
        for (uint64_t i = 1; i <= N; i++) {
            uint32_t steps = collatz_steps(i);
            sum += steps;

            if (steps > max)
                max = steps;
        }

    } else if (schedule_type == 3) {

        #pragma omp parallel for schedule(dynamic,10000) reduction(+:sum) reduction(max:max)
        for (uint64_t i = 1; i <= N; i++) {
            uint32_t steps = collatz_steps(i);
            sum += steps;

            if (steps > max)
                max = steps;
        }

    } else {

        #pragma omp parallel for schedule(guided) reduction(+:sum) reduction(max:max)
        for (uint64_t i = 1; i <= N; i++) {
            uint32_t steps = collatz_steps(i);
            sum += steps;

            if (steps > max)
                max = steps;
        }
    }

    double end = omp_get_wtime();

    *checksum = sum % MOD;
    *max_steps = max;

    return end - start;
}

/* ---------------------------------------------------------
   Phase 4B
   --------------------------------------------------------- */
static void run_phase4B(FILE *csv, int max_threads)
{
    printf("\n========== PHASE 4B: SCHEDULING ==========\n");

    fprintf(csv,
            "\nPHASE4B,schedule,chunk,threads,"
            "run1,run2,run3,avg_time,throughput\n");

    const char *names[] = {
        "static",
        "static_1000",
        "dynamic_100",
        "dynamic_10000",
        "guided"
    };

    int types[] = {0, 1, 2, 3, 4};
    int chunks[] = {0, 1000, 100, 10000, 0};

    for (int x = 0; x < 5; x++) {

        uint64_t checksum;
        uint32_t max_steps;

        /*
         * Warmup
         */
        double run1 =
            schedule_test(
                max_threads,
                types[x],
                chunks[x],
                &checksum,
                &max_steps
            );

        /*
         * Run 2
         */
        double run2 =
            schedule_test(
                max_threads,
                types[x],
                chunks[x],
                &checksum,
                &max_steps
            );

        /*
         * Run 3
         */
        double run3 =
            schedule_test(
                max_threads,
                types[x],
                chunks[x],
                &checksum,
                &max_steps
            );

        double avg =
            (run2 + run3) / 2.0;

        double throughput =
            (double)N / avg;

        printf("\nSchedule: %s\n", names[x]);
        printf("Run 1: %.6f s\n", run1);
        printf("Run 2: %.6f s\n", run2);
        printf("Run 3: %.6f s\n", run3);
        printf("Average: %.6f s\n", avg);
        printf("Throughput: %.2f iter/sec\n",
               throughput);

        fprintf(csv,
                "PHASE4B,%s,%d,%d,"
                "%.9f,%.9f,%.9f,%.9f,%.3f\n",
                names[x],
                chunks[x],
                max_threads,
                run1,
                run2,
                run3,
                avg,
                throughput);
    }
}

/* ---------------------------------------------------------
   MAIN
   --------------------------------------------------------- */
int main(void)
{
    printf("=================================================\n");
    printf(" Zeba Academy - The Amdahl Reality Gap\n");
    printf(" Student ID: 230103057\n");
    printf(" Workload N: %" PRIu64 "\n", N);
    printf("=================================================\n");

    printf("\nOpenMP maximum threads: %d\n",
           omp_get_max_threads());

    printf("OpenMP processors: %d\n",
           omp_get_num_procs());

    /*
     * This should be 8 on your i3-1215U.
     */
    int max_threads = omp_get_max_threads();

    if (max_threads > 8)
        max_threads = 8;

    printf("Maximum threads used in this lab: %d\n",
           max_threads);

    /*
     * Open results.csv
     */
    FILE *csv = fopen("results.csv", "w");

    if (csv == NULL) {
        perror("Cannot create results.csv");
        return EXIT_FAILURE;
    }

    fprintf(csv,
            "# Student ID,230103057\n"
            "# N,%" PRIu64 "\n"
            "# Compiler flags,-O2 -fopenmp\n"
            "# Timing,omp_get_wtime\n"
            "# Run 1 is warmup and excluded from averages\n\n",
            N);

    /*
     * ------------------------------------------------------
     * PHASE 2
     * ------------------------------------------------------
     */

    uint64_t checksum;
    uint32_t max_steps;
    uint64_t total_hits;

    double T_seq =
        benchmark_sequential(
            &checksum,
            &max_steps,
            &total_hits
        );

    fprintf(csv,
            "PHASE2,threads,run1,run2,run3,"
            "T_seq,checksum,max_steps,hits\n");

    /*
     * We don't retain Run1/2/3 separately here,
     * because benchmark_sequential already printed them.
     *
     * T_seq is the important value.
     */
    fprintf(csv,
            "PHASE2,1,NA,NA,NA,%.9f,"
            "%" PRIu64 ",%" PRIu32 ",%" PRIu64 "\n",
            T_seq,
            checksum,
            max_steps,
            total_hits);

    /*
     * ------------------------------------------------------
     * PHASE 3
     * ------------------------------------------------------
     */

    run_phase3(
        csv,
        T_seq,
        checksum
    );

    /*
     * ------------------------------------------------------
     * PHASE 4A
     * ------------------------------------------------------
     */

    run_phase4A(
        csv,
        max_threads
    );

    /*
     * ------------------------------------------------------
     * PHASE 4B
     * ------------------------------------------------------
     */

    run_phase4B(
        csv,
        max_threads
    );

    fclose(csv);

    /*
     * Final verification
     */
    printf("\n=================================================\n");
    printf(" FINAL VERIFICATION\n");
    printf("=================================================\n");

    printf("Student ID:       230103057\n");
    printf("N:                %" PRIu64 "\n", N);
    printf("Checksum:         %" PRIu64 "\n", checksum);
    printf("Maximum steps:    %" PRIu32 "\n", max_steps);
    printf("Hits >100:        %" PRIu64 "\n", total_hits);
    printf("T_seq:            %.6f s\n", T_seq);
    printf("Results file:     results.csv\n");

    printf("\nAll experiments completed.\n");
    printf("=================================================\n");

    return 0;
}