/*
 * Congruence-Only Sieve for Prime Numbers in Quadratic Intervals
 * CPU Implementation for Fair GPU vs CPU Comparison
 * Authors: S. Tabalevich, S. Aleksandrov
 * Email: mastakby@ya.ru
 * 
 * This implementation provides fair CPU baseline for GPU comparison
 * Complexity: O(N log log N) time, O(N) bit memory
 * 
 * MIT License
 * 
 * Copyright (c) 2025 S. Tabalevich, S. Aleksandrov
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

// Generate all primes ≤ N using simple sieve
void generate_primes_simple(int N, bool **primes, int *prime_count) {
    *primes = (bool*)calloc(N + 1, sizeof(bool));
    (*primes)[0] = (*primes)[1] = false;
    for (int i = 2; i <= N; i++) (*primes)[i] = true;
    
    *prime_count = 0;
    for (int p = 2; p * p <= N; p++) {
        if ((*primes)[p]) {
            for (int multiple = p * p; multiple <= N; multiple += p) {
                (*primes)[multiple] = false;
            }
        }
    }
    
    for (int i = 2; i <= N; i++) {
        if ((*primes)[i]) (*prime_count)++;
    }
}

// CPU implementation of congruence-only sieve for I_N
int congruence_sieve_cpu(int N, bool *primes, int *primes_found) {
    int interval_size = N * N - (N - 2) * (N - 2);
    bool *sieve = (bool*)malloc(interval_size * sizeof(bool));
    
    // Initialize all positions as potential primes
    for (int i = 0; i < interval_size; i++) {
        sieve[i] = true;
    }
    
    int offset = (N - 2) * (N - 2) + 1;
    int found_count = 0;
    
    // Sieve using congruence-only approach
    for (int p = 2; p <= N; p++) {
        if (!primes[p]) continue;
        
        int r = (N * N) % p;
        int pos = (r - (offset % p) + p) % p;
        
        for (int i = pos; i < interval_size; i += p) {
            sieve[i] = false;
        }
    }
    
    // Count primes in interval
    *primes_found = 0;
    for (int i = 0; i < interval_size; i++) {
        if (sieve[i]) {
            int prime = offset + i;
            if (prime > 1) {  // Skip offset position
                (*primes_found)++;
            }
        }
    }
    
    free(sieve);
    return *primes_found;
}

// Benchmark function
double benchmark_congruence_sieve_cpu(int N, int iterations) {
    bool *primes = NULL;
    int prime_count = 0;
    clock_t start, end;
    
    start = clock();
    for (int i = 0; i < iterations; i++) {
        int found = 0;
        
        // Generate primes ≤ N (preprocessing)
        generate_primes_simple(N, &primes, &prime_count);
        
        // Perform congruence sieve
        congruence_sieve_cpu(N, primes, &found);
        
        free(primes);
    }
    end = clock();
    
    return ((double)(end - start)) / CLOCKS_PER_SEC / iterations;
}

// Main benchmark program
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <N> [iterations]\n", argv[0]);
        printf("Example: %s 1000000 10\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    int iterations = (argc > 2) ? atoi(argv[2]) : 5;
    
    printf("CPU Congruence-Only Sieve Benchmark\n");
    printf("====================================\n");
    printf("Problem size (N): %d\n", N);
    printf("Interval: I_N = ((N-2)^2, N^2)\n");
    printf("Interval size: %d\n", N * N - (N - 2) * (N - 2));
    printf("Iterations: %d\n", iterations);
    
    double avg_time = benchmark_congruence_sieve_cpu(N, iterations);
    
    printf("\nResults:\n");
    printf("Average time: %.6f seconds\n", avg_time);
    printf("Throughput: %.2f operations/second\n", 1.0 / avg_time);
    
    return 0;
}