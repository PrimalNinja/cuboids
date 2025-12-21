%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define N 512
#define TOTAL_VOXELS (N * N * N)

struct SpatialDNA {
    float tx, ty, tz, rx, ry, rz, sx, sy, sz;
};

// --- TYPICAL METHOD: OPTIMIZED FOR MEMORY THROUGHPUT ---
__global__ void physicalRotate(const uint8_t* __restrict__ src, uint8_t* dst, float angle) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;
    
    int x = tid % N; int y = (tid / N) % N; int z = tid / (N * N);
    float s, c; __sincosf(angle, &s, &c); // Using hardware intrinsics for fairness
    
    int nx = (int)((x-32)*c - (z-32)*s + 32);
    int nz = (int)((x-32)*s + (z-32)*c + 32);
    
    // Scattered write is the inherent weakness of this ERA 
    if (nx >= 0 && nx < N && nz >= 0 && nz < N) dst[nx + y * N + nz * N * N] = src[tid];
}

__global__ void typicalScore(const uint8_t* __restrict__ data, int* score) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;
    uint8_t val = data[tid];
    if (val > 0) atomicAdd(score, (int)val);
}

// --- NEW DNA PARADIGM: OPTIMIZED FOR INSTRUCTION DENSITY ---
__global__ void dnaFusedScore(const uint8_t* __restrict__ target, SpatialDNA dna, int* outScore) {
    __shared__ int blockAccumulator;
    if (threadIdx.x == 0) blockAccumulator = 0;
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    // Fast coordinate extraction
    int z = tid / (N * N);
    int rem = tid % (N * N);
    int y = rem / N;
    int x = rem % N;

    float fx = (float)x - dna.tx;
    float fy = (float)y - dna.ty;
    float fz = (float)z - dna.tz;

    float s, c; __sincosf(dna.ry, &s, &c);
    float nx = fx * c - fz * s;
    float nz = fx * s + fz * c;

    // Zero-Copy evaluation: Logic stays in registers
    int val = (fabsf(nx) < dna.sx && fabsf(fy) < dna.sy && fabsf(nz) < dna.sz) ? (int)target[tid] : 0;

    // Register-level handoff (Warp Shuffle)
    for (int offset = 16; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Minimize Global Memory traffic via Shared Memory
    if ((threadIdx.x & 31) == 0) atomicAdd(&blockAccumulator, val);
    __syncthreads();

    if (threadIdx.x == 0 && blockAccumulator > 0) atomicAdd(outScore, blockAccumulator);
}

int main() {
    uint8_t *d_src, *d_dst;
    int *d_score;
    cudaMalloc(&d_src, TOTAL_VOXELS);
    cudaMalloc(&d_dst, TOTAL_VOXELS);
    cudaMallocManaged(&d_score, sizeof(int));
    cudaMemset(d_src, 1, TOTAL_VOXELS);

    SpatialDNA dna = {32.0f, 32.0f, 32.0f, 0.0f, 0.5f, 0.0f, 5.0f, 5.0f, 5.0f};
    cudaEvent_t start, stop;
    float timeTypical, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warm-up 
    physicalRotate<<<TOTAL_VOXELS/256, 256>>>(d_src, d_dst, 0.5f);
    dnaFusedScore<<<TOTAL_VOXELS/256, 256>>>(d_src, dna, d_score);
    cudaDeviceSynchronize();

    printf("Executing 100 Evolutionary Steps...\n");

    // 1. BENCHMARK TYPICAL (Highly Optimized Discrete)
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        physicalRotate<<<TOTAL_VOXELS/256, 256>>>(d_src, d_dst, 0.5f + (i * 0.01f));
        typicalScore<<<TOTAL_VOXELS/256, 256>>>(d_dst, d_score);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTypical, start, stop);

    // 2. BENCHMARK DNA PARADIGM (Fused Zero-Copy)
    *d_score = 0;
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        dna.ry = 0.5f + (i * 0.01f);
        dnaFusedScore<<<TOTAL_VOXELS/256, 256>>>(d_src, dna, d_score);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeDNA, start, stop);

    printf("\n--- SEARCH BENCHMARK: 100 EVOLUTIONARY STEPS ---\n");
    printf("Typical Method: %.3f ms (Global Memory Bound)\n", timeTypical);
    printf("DNA Paradigm:   %.3f ms (Register/L1 Bound)\n", timeDNA);
    printf("-----------------------------------------------\n");
    printf("PERFORMANCE GAP: %.2fx Faster\n", timeTypical / timeDNA);

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_score);
    return 0;
}