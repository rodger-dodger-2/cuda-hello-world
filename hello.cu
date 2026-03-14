#include <stdio.h>

__global__ void hello_kernel(int total_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x * threadIdx.x;
    printf("  [thread %2d / %d] Hello from the GPU!\n", tid, total_threads);
}

int main() {
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== GPU Info ===\n");
    printf("  Device:            %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Multiprocessors:    %d\n", prop.multiProcessorCount);
    printf("  Total global mem:   %.0f MB\n", (float)prop.totalGlobalMem / 1024 / 1024);
    printf("\n=== Kernel Output ===\n");

    int blocks = 2;
    int threads_per_block = 8;
    hello_kernel<<<blocks, threads_per_block>>>(blocks * threads_per_block);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\n✓ CUDA hello world succeeded on %s\n", prop.name);
    return 0;
}
