#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>

// Warp-level reduction using shuffle instructions — no shared memory needed
// within a warp, registers communicate directly across the 32 threads.
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Each block grid-strides the array loading float4 (4 floats per instruction),
// accumulates locally, then reduces with warp shuffles. Only 32 shared-memory
// slots are needed (one per warp leader) instead of one per thread.
__global__ void reduce_shuffle(const float4 *input, float *block_sums, int n4) {
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
        float4 v = input[i];
        acc += v.x + v.y + v.z + v.w;
    }
    acc = warp_reduce(acc);

    __shared__ float sdata[32];   // max 32 warps per block
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) sdata[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        acc = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        acc = warp_reduce(acc);
        if (lane == 0) block_sums[blockIdx.x] = acc;
    }
}

__global__ void reduce_pass2(const float *bs, float *result, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = (tid < n) ? bs[tid] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) result[0] = sdata[0];
}

int main() {
    const int N = 100000000, N4 = N / 4, THREADS = 256, BLOCKS = 1024;
    const size_t bytes = N * sizeof(float);

    float *h = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h[i] = 1.0f;

    float *d_in, *d_bs, *d_res;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_bs, BLOCKS * sizeof(float));
    cudaMalloc(&d_res, sizeof(float));
    cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);

    // CUB workspace
    void *d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_in, d_res, N);
    cudaMalloc(&d_tmp, tmp_bytes);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;

    // ── Warp-shuffle + float4 ─────────────────────────────────────────────────
    cudaEventRecord(s);
    reduce_shuffle<<<BLOCKS, THREADS>>>((float4 *)d_in, d_bs, N4);
    reduce_pass2<<<1, BLOCKS, BLOCKS * sizeof(float)>>>(d_bs, d_res, BLOCKS);
    cudaEventRecord(e); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, s, e);
    float r1; cudaMemcpy(&r1, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    printf("=== Warp-Shuffle + float4 ===\n");
    printf("  Result: %.0f  time: %.3f ms  bandwidth: %.1f GB/s  %s\n\n",
           r1, ms, bytes / (ms / 1000.0) / 1e9, ((long long)r1 == N) ? "correct" : "WRONG");

    // ── CUB DeviceReduce (library best-in-class) ──────────────────────────────
    cudaEventRecord(s);
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_in, d_res, N);
    cudaEventRecord(e); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, s, e);
    float r2; cudaMemcpy(&r2, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    printf("=== CUB DeviceReduce::Sum ===\n");
    printf("  Result: %.0f  time: %.3f ms  bandwidth: %.1f GB/s  %s\n\n",
           r2, ms, bytes / (ms / 1000.0) / 1e9, ((long long)r2 == N) ? "correct" : "WRONG");

    printf("  T4 peak bandwidth: ~320 GB/s\n");

    free(h);
    cudaFree(d_in); cudaFree(d_bs); cudaFree(d_res); cudaFree(d_tmp);
    return 0;
}
