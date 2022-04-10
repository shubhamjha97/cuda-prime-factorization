#include<iostream>
#include <cuda.h>
#include <stdlib.h>
using namespace std;

// TODO: Put the frequency array on shared mem
// TODO: Check limit <=

void  factorize_kernel(char *, unsigned int);
void outputResult(unsigned int, char *);

int main(int argc, char* argv[]) {
    unsigned int n = stoi(argv[1]);

    // Calculate the number of times n is divisible by 2
    unsigned int two_freq = 0;
    while(!(n%2)) {
        two_freq++;
        n/=2;
    }

    // Initialize some variables
    unsigned int limit = ceil(sqrt(n));
    unsigned int frequency_table_size = limit/2;
    unsigned int frequency_table_bytes = frequency_table_size * sizeof(char);

    // Initialize frequency table
    char *freq_table_gpu; // Use char so that the array fits in cache
    char *frequency_table = (char *)calloc(, sizeof(char));
    if(!frequency_table) {
        fprintf(stderr, " Cannot allocate the 1x%u array\n", N);
        exit(1);
    }

    // Copy frequency table to device
    cudaMemcpy(
            frequency_table_gpu,
            frequency_table,
            frequency_table_bytes,
            cudaMemcpyHostToDevice);

    // Start the kernel
    int threads = 1024, blocks = ceil(limit/1024);
    factorize_kernel<<<blocks, threads>>>(frequency_table, n);

    // Copy result back to host
    cudaMemcpy(
            frequency_table,
            freq_table_gpu,
            frequency_table_bytes,
            cudaMemcpyDeviceToHost);
    cudaFree(frequency_table_gpu);

    // Output result to console
    processResult(two_freq, frequency_table);

    return 0;
}

void printInteger(unsigned int i) {
    printf("%d ", x);
}

void processResult(unsigned int two_freq, char *frequency_table, unsigned int limit) {
    while(two_freq) {
        oputput(2);
    }
    for(int i=0; i<=limit; i++) {
        if(frequency_table[i]) {
            printInteger(3 + 2 * i);
        }
    }
}

__global__
void factorize_kernel(char *frequency_table, unsigned int n) {
    unsigned int i = 3 + 2 * threadIdx.x; // TODO: try doing a bit shift instead of multiplication
    unsigned int freq = 0;
    while(!(n%i)) {
        freq++;
        n/=i;
    }
    if(freq) frequency_table[threadIdx.x] = freq;
}
