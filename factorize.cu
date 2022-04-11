#define BLOCK_SIZE 1024
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
using namespace std;

// TODO: Put the frequency array on shared mem

__global__
void factorize_kernel(char *, unsigned int, unsigned int);
void processResult(unsigned int, unsigned int, char *, unsigned int);
void printUnsignedInteger(unsigned int);

int main(int argc, char* argv[]) {
    unsigned int n;
    sscanf(argv[1], "%u", &n);

    // Calculate the number of times n is divisible by 2
    unsigned int two_freq = 0;
    while(!(n%2)) {
        two_freq++;
        n/=2;
    }

    // Initialize frequency table
    unsigned int limit = ceil(sqrt(n));
    unsigned int frequency_table_size = (limit-1)/2;
    unsigned int frequency_table_bytes = frequency_table_size * sizeof(char);

    char *frequency_table_gpu; // Use char so that the array fits in cache
    char *frequency_table = (char *)calloc(frequency_table_size, sizeof(char));
    if(!frequency_table) {
        fprintf(stderr, "Cannot allocate the 1x%u frequency table.\n", frequency_table_size);
        exit(1);
    }
    
    // Copy frequency table to device
    cudaMalloc((void**)&frequency_table_gpu, frequency_table_bytes);
    cudaMemcpy(
            frequency_table_gpu,
            frequency_table,
            frequency_table_bytes,
            cudaMemcpyHostToDevice);

    // Start the kernel
    int threads = BLOCK_SIZE, blocks = ceil((double)frequency_table_size/(threads));
    factorize_kernel<<<blocks, threads>>>(frequency_table_gpu, n, frequency_table_size);

    // Copy result back to host
    cudaMemcpy(
            frequency_table,
            frequency_table_gpu,
            frequency_table_bytes,
            cudaMemcpyDeviceToHost);
    cudaFree(frequency_table_gpu);

    // Output result to console
    processResult(n, two_freq, frequency_table, frequency_table_size);

    return 0;
}

void printUnsignedInteger(unsigned int i) {
    printf("%u ", i);
}

void processResult(unsigned int n, unsigned int two_freq, char *frequency_table, unsigned int frequency_table_size) {
    unsigned int temp = 1;
    while(two_freq--) {
        printUnsignedInteger(2);
    }

    for(unsigned int i=0; i<frequency_table_size; i++) {
        if(frequency_table[i]) {
            while(frequency_table[i]--) {
                int current = 3 + 2 * i;
                printUnsignedInteger(current);
                temp *= current;
            }
        }
    }

    // Prime number case
    if(temp!=n) {
        printUnsignedInteger(n);
    }
}

__global__
void factorize_kernel(char *frequency_table, unsigned int n, unsigned int freq_table_size) {
    if(threadIdx.x < freq_table_size) {
        unsigned int i = 3 + (2 * threadIdx.x); // TODO: try doing a bit shift instead of multiplication
        unsigned int freq = 0;
        while(!(n%i)) {
            freq++;
            n/=i;
        }
        if(freq) frequency_table[BLOCK_SIZE*blockIdx.x + threadIdx.x] = freq;
    }
}
