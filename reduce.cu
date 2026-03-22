#include <stdio.h>
#include <stdlib.h>

// Each block reduces its chunk of the array into a single value using shared
// memory. Threads first load their element, then repeatedly fold the upper half
// of active threads into the lower half until one value remains per block.
__global__ void reduce_sum(const float *input, float *block_sums, int n) {
    extern __shared__ float sdata[];

    int tid   = threadIdx.x;
    int gid   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        block_sums[blockIdx.x] = sdata[0];
}

int main() {
    const int N              = 1 << 20;   // 1 M elements
    const int THREADS        = 256;
    const int BLOCKS         = (N + THREADS - 1) / THREADS;
    const size_t bytes       = N * sizeof(float);
    const size_t block_bytes = BLOCKS * sizeof(float);

    // --- host data ---
    float *h_input     = (float *)malloc(bytes);
    float *h_block_sum = (float *)malloc(block_bytes);

    // fill with 1.0 so the expected sum is exactly N
    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    // --- device data ---
    float *d_input, *d_block_sums;
    cudaMalloc(&d_input,      bytes);
    cudaMalloc(&d_block_sums, block_bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // --- run kernel ---
    size_t shared_mem = THREADS * sizeof(float);
    reduce_sum<<<BLOCKS, THREADS, shared_mem>>>(d_input, d_block_sums, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // finish reduction on the host (block_sums is tiny)
    cudaMemcpy(h_block_sum, d_block_sums, block_bytes, cudaMemcpyDeviceToHost);

    double gpu_sum = 0.0;
    for (int i = 0; i < BLOCKS; i++)
        gpu_sum += h_block_sum[i];

    printf("=== Parallel Reduction (sum) ===\n");
    printf("  Elements : %d (%.1f M)\n", N, N / 1e6);
    printf("  Blocks   : %d  Threads/block : %d\n", BLOCKS, THREADS);
    printf("  GPU sum  : %.0f\n", gpu_sum);
    printf("  Expected : %d\n", N);

    int ok = ((long long)gpu_sum == N);
    printf("\n%s parallel reduction on GPU\n", ok ? "✓" : "✗");

    free(h_input);
    free(h_block_sum);
    cudaFree(d_input);
    cudaFree(d_block_sums);

    return ok ? 0 : 1;
}
