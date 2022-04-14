#define BLOCK_SIZE 1024
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
using namespace std;

// Function declarations
void processResult(unsigned int, unsigned int, unsigned int *, char *, unsigned int);
void printUnsignedInteger(unsigned int);
__global__
void sieve_kernel(bool *, unsigned int, unsigned int, unsigned int);
__global__
void factorize_kernel(unsigned int *, char *, unsigned int, unsigned int);
unsigned int count_two_divisible(unsigned int&);
bool* generate_sieve(unsigned int, unsigned int&);
unsigned int *generate_prime_numbers_array(bool*, unsigned int, unsigned int&);
char* prime_factorization(unsigned int *, unsigned int, unsigned int);

int main(int argc, char* argv[]) {
    unsigned int n;
    sscanf(argv[1], "%u", &n);

    // Calculate the number of times n is divisible by 2
    unsigned int two_freq = count_two_divisible(n);

    // Generate Eratosthenes Sieve
    unsigned int sieve_size = 0;
    bool *sieve = generate_sieve(n, sieve_size);

    // Generate array of prime numbers
    unsigned int prime_number_array_size = 0;
    unsigned int *prime_number_array = generate_prime_numbers_array(sieve, sieve_size, prime_number_array_size);

    // Generate prime factorization
    char *prime_factors_freq = prime_factorization(prime_number_array, prime_number_array_size, n);

    // Process and print the result
    processResult(n, two_freq, prime_number_array, prime_factors_freq, prime_number_array_size);

    return 0;
}

void printUnsignedInteger(unsigned int i) {
    printf("%u ", i);
}

void processResult(unsigned int n, unsigned int two_freq, unsigned int *prime_number_array, char *frequency_table, unsigned int frequency_table_size) {
    unsigned int temp = 1;
    while(two_freq--) {
        printUnsignedInteger(2);
    }

    for(unsigned int i=1; i<frequency_table_size; i++) {
        if(frequency_table[i]) {
            while(frequency_table[i]--) {
                int current = prime_number_array[i];
                printUnsignedInteger(current);
                temp *= current;
            }
        }
    }

    // Prime number case
    if(temp!=n) {
        printUnsignedInteger(n/temp);
    }
}

unsigned int count_two_divisible(unsigned int &n) {
    unsigned int two_freq = 0;
    while(!(n%2)) {
        two_freq++;
        n/=2;
    }
    return two_freq;
}

bool* generate_sieve(unsigned int n, unsigned int &sieve_size) {
    // Initialize sieve array
    unsigned int limit = ceil(sqrt(n));
    sieve_size = ceil((double)limit/2);
    unsigned int sieve_bytes = sieve_size * sizeof(bool);

    bool *sieve = (bool *)calloc(sieve_size, sizeof(bool));
    if(!sieve) {
        fprintf(stderr, "Cannot allocate the 1x%u sieve.\n", sieve_size);
        exit(1);
    }
    for(int i=0; i<sieve_size; i++) {
        sieve[i] = true;
    }

    // Copy sieve to GPU
    bool *sieve_gpu;
    cudaMalloc((void**)&sieve_gpu, sieve_bytes);
    cudaMemcpy(
            sieve_gpu,
            sieve,
            sieve_bytes,
            cudaMemcpyHostToDevice);

    // Start the kernel
    int threads = BLOCK_SIZE, blocks = ceil((double)sieve_size/(threads));
    sieve_kernel<<<blocks, threads>>>(sieve_gpu, sieve_size, n, limit);

    // Copy sieve to host
    cudaMemcpy(
            sieve,
            sieve_gpu,
            sieve_bytes,
            cudaMemcpyDeviceToHost);
    cudaFree(sieve_gpu);

    return sieve;
}

unsigned int *generate_prime_numbers_array(bool *sieve, unsigned int sieve_size, unsigned int &prime_number_array_size) {
    prime_number_array_size = 0;
    for(int i=0; i<sieve_size; i++) {
        if(sieve[i]) {
            prime_number_array_size++;
        }
    }
    unsigned int *prime_number_array = (unsigned int *)calloc(prime_number_array_size, sizeof(unsigned int));
    int p_idx = 0;
    for(int i=0; i<sieve_size; i++) {
        if(sieve[i]) {
            prime_number_array[p_idx++] = 1 + 2*i;
        }
    }

    return prime_number_array;
}

char *prime_factorization(unsigned int *prime_number_array, unsigned int array_size, unsigned int n) {
    // Initialize frequency table on host
    unsigned int frequency_table_bytes = array_size * sizeof(char);
    unsigned int prime_number_array_bytes = array_size * sizeof(unsigned int);

    char *frequency_table = (char *)calloc(array_size, sizeof(char));
    if(!frequency_table) {
        fprintf(stderr, "Cannot allocate the 1x%u frequency table.\n", array_size);
        exit(1);
    }

    // Copy arrays to device
    unsigned int *prime_number_array_gpu;
    char *frequency_table_gpu;
    cudaMalloc((void**)&frequency_table_gpu, frequency_table_bytes);
    cudaMemcpy(frequency_table_gpu, frequency_table, frequency_table_bytes, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&prime_number_array_gpu, prime_number_array_bytes);
    cudaMemcpy(prime_number_array_gpu, prime_number_array, prime_number_array_bytes, cudaMemcpyHostToDevice);

    // Start the kernel
    int threads = BLOCK_SIZE, blocks = ceil((double)array_size/(threads));
    factorize_kernel<<<blocks, threads>>>(prime_number_array_gpu, frequency_table_gpu, n, array_size);

    // Copy result back to host
    cudaMemcpy(frequency_table, frequency_table_gpu, frequency_table_bytes, cudaMemcpyDeviceToHost);
    cudaFree(frequency_table_gpu);
    cudaFree(prime_number_array_gpu);

    return frequency_table;
}

__global__
void factorize_kernel(unsigned int *prime_numbers, char *frequency_table, unsigned int n, unsigned int array_size) {
    unsigned int global_thread_index = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if(global_thread_index > 0 && global_thread_index < array_size) {
        unsigned int i = prime_numbers[global_thread_index];
        unsigned int freq = 0;
        while(!(n%i)) {
            freq++;
            n/=i;
        }
        if(freq) frequency_table[global_thread_index] = freq;
    }
}

__global__
void sieve_kernel(bool *sieve_gpu, unsigned int sieve_size, unsigned int n, unsigned int limit) {
    unsigned int global_thread_index = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if(global_thread_index > 0 && global_thread_index < sieve_size) {
        unsigned int p = 1 + (2 * global_thread_index);
        unsigned int step_size = 2*p;
        for (unsigned int i = p * p; i <= limit; i += step_size) {
            int idx = (i - 1) / 2;
            sieve_gpu[idx] = false;
        }
    }
}
