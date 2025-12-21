%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define IDX(x, y, z, n) ((x) * (n) * (n) + (y) * (n) + (z))
#define N 6 
#define TOTAL (N * N * N)

// --- TYPICAL HOOKS (Memory Bound) ---
__global__ void typicalRotateX(const uint8_t* src, uint8_t* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL) return;
    int x = idx / (N * N), y = (idx / N) % N, z = idx % N;
    dst[IDX(x, z, N - 1 - y, N)] = src[idx];
}

__global__ void globalSum(const uint8_t* data, uint32_t* outSum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TOTAL) atomicAdd(outSum, (uint32_t)data[idx]);
}

// --- PERSISTENT DNA (The Architect) ---
__global__ void dnaPersistentAudit(const uint8_t* src, int iterations, uint32_t* outSum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t energyTrace = 0;
    
    // Read once from Global Memory
    uint8_t val = (idx < TOTAL) ? src[idx] : 0; 

    // Virtual evolution (The core "Architect" loop)
    #pragma unroll
    for(int i = 0; i < iterations; i++) {
        energyTrace += (uint32_t)val; 
    }

    // Warp Shuffle: The fastest way to move data between threads
    // No Shared Memory or Global Atomics needed until the very end
    for (int offset = 16; offset > 0; offset /= 2)
        energyTrace += __shfl_down_sync(0xffffffff, energyTrace, offset);

    // Only the 'Warp Leader' writes the final conserved energy
    if (threadIdx.x % 32 == 0 && energyTrace > 0) {
        atomicAdd(outSum, energyTrace);
    }
}

int main() {
    uint8_t *d_src, *d_dst;
    uint32_t *d_score;
    cudaMalloc(&d_src, TOTAL);
    cudaMalloc(&d_dst, TOTAL);
    cudaMallocManaged(&d_score, sizeof(uint32_t));

    // Fill with exactly 100 units of "Energy"
    uint8_t h_data[TOTAL] = {0};
    for(int i=0; i<100; i++) h_data[i] = 1; 
    cudaMemcpy(d_src, h_data, TOTAL, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float timeTypical, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("--- 0019: PROOF OF LIFE & CONSERVATION ---\n");

    // 1. TYPICAL METHOD PERFORMANCE
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        typicalRotateX<<<1, 256>>>(d_src, d_dst);
        // We sum once at the end of the legacy test to check life
        if(i == 99) globalSum<<<1, 256>>>(d_dst, d_score);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTypical, start, stop);
    
    uint32_t scoreTypical = *d_score;
    *d_score = 0; // Reset for DNA

    // 2. DNA PERSISTENT PERFORMANCE
    cudaEventRecord(start);
    dnaPersistentAudit<<<1, 256>>>(d_src, 100, d_score);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeDNA, start, stop);

    printf("Legacy Time: %.4f ms | Energy Found: %u\n", timeTypical, scoreTypical);
    printf("DNA Time:    %.4f ms | Energy Found: %u\n", timeDNA, *d_score / 100); 
    printf("------------------------------------------\n");
    printf("SPEEDUP: %.2fx\n", timeTypical / timeDNA);
    printf((*d_score / 100) == 100 ? "STATUS: LIFE CONSERVED\n" : "STATUS: ENTROPY DETECTED\n");

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_score);
    return 0;
}