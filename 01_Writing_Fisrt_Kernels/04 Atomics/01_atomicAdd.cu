#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

__global__ void incrementCounterNonAtomic(int* counter) {
    int old = *counter;
    int new_value = old + 1;
    *counter = new_value;
}

__global__ void incrementCounterAtomic(int* counter) {
    int a = atomicAdd(counter, 1);
}

int main() {
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    int *d_counterNonAtomic, *d_counterAtomic;

    cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterAtomic, sizeof(int));

    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);

    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);

    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Non-atomic counter value: %d\n", h_counterNonAtomic);
    printf("Atomic counter value: %d\n", h_counterAtomic);

    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);

    return 0;
}