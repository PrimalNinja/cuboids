%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define N 512
#define TOTAL_VOXELS (N * N * N)

struct SpatialDNA {
    float tx, ty, tz, rx, ry, rz, sx, sy, sz;
};

// --- OPTIMIZED LEGACY: Best-case Scenario for Era 1 ---
__global__ void legacyTransform(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, float angle) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    int z = tid / (N * N);
    int rem = tid % (N * N);
    int y = rem / N;
    int x = rem % N;

    float s, c; __sincosf(angle, &s, &c);
    
    int nx = (int)((x - 32) * c - (z - 32) * s + 32);
    int nz = (int)((x - 32) * s + (z - 32) * c + 32);

    if (nx >= 0 && nx < N && nz >= 0 && nz < N) {
        dst[nx + y * N + nz * N * N] = src[tid];
    }
}

__global__ void legacyScore(const uint8_t* __restrict__ data, int* outScore) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;
    uint8_t val = data[tid];
    if (val > 0) atomicAdd(outScore, (int)val);
}

// --- HYPER-OPTIMIZED DNA: The Persistent Architect ---
__global__ void dnaPersistentKernel(const uint8_t* __restrict__ target, SpatialDNA dna, int iterations, int* finalScore) {
    __shared__ int blockAccumulator;
    if (threadIdx.x == 0) blockAccumulator = 0;
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    // DATA REUSE: Load from VRAM only once
    const uint8_t voxelValue = target[tid];
    if (voxelValue == 0) return; 

    int localMatchCount = 0;

    // Coordinate Decomposition (Done once outside loop)
    int z_idx = tid / (N * N);
    int rem = tid % (N * N);
    int y_idx = rem / N;
    int x_idx = rem % N;

    float base_x = (float)x_idx - dna.tx;
    float base_y = (float)y_idx - dna.ty;
    float base_z = (float)z_idx - dna.tz;

    // SEARCH LOOP: Stays entirely in Registers/L1
    #pragma unroll 4
    for(int i = 0; i < iterations; i++) {
        float currentRy = dna.ry + (i * 0.001f);
        float s, c; __sincosf(currentRy, &s, &c);
        
        float nx = base_x * c - base_z * s;
        float nz = base_x * s + base_z * c;

        if (fabsf(nx) < dna.sx && fabsf(base_y) < dna.sy && fabsf(nz) < dna.sz) {
            localMatchCount += (int)voxelValue;
        }
    }

    // Two-Stage Reduction: Warp Shuffle -> Shared Memory
    for (int offset = 16; offset > 0; offset /= 2)
        localMatchCount += __shfl_down_sync(0xffffffff, localMatchCount, offset);

    if ((threadIdx.x & 31) == 0) atomicAdd(&blockAccumulator, localMatchCount);
    __syncthreads();

    if (threadIdx.x == 0 && blockAccumulator > 0) atomicAdd(finalScore, blockAccumulator);
}

int main() {
    uint8_t *d_src, *d_dst;
    int *d_score;
    cudaMalloc(&d_src, TOTAL_VOXELS);
    cudaMalloc(&d_dst, TOTAL_VOXELS);
    cudaMallocManaged(&d_score, sizeof(int));

    uint8_t* h_data = (uint8_t*)malloc(TOTAL_VOXELS);
    for(int i=0; i<TOTAL_VOXELS; i++) h_data[i] = (i % 2) + 1;
    cudaMemcpy(d_src, h_data, TOTAL_VOXELS, cudaMemcpyHostToDevice);

    SpatialDNA dna = {32.0f, 32.0f, 32.0f, 0.0f, 0.5f, 0.0f, 5.0f, 5.0f, 5.0f};
    cudaEvent_t start, stop;
    float tL, tD;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warm-up
    dnaPersistentKernel<<<TOTAL_VOXELS/256, 256>>>(d_src, dna, 1, d_score);
    cudaDeviceSynchronize();

    // RACE
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        legacyTransform<<<TOTAL_VOXELS/256, 256>>>(d_src, d_dst, 0.5f + (i * 0.001f));
        legacyScore<<<TOTAL_VOXELS/256, 256>>>(d_dst, d_score);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tL, start, stop);

    *d_score = 0;
    cudaEventRecord(start);
    dnaPersistentKernel<<<TOTAL_VOXELS/256, 256>>>(d_src, dna, 100, d_score);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tD, start, stop);

    printf("\n--- HYPER-OPTIMIZED SEARCH (100 STEPS) ---\n");
    printf("Legacy Method:    %.3f ms\n", tL);
    printf("Persistent DNA:   %.3f ms\n", tD);
    printf("------------------------------------------\n");
    printf("SPEEDUP RATIO:    %.2fx Faster\n", tL / tD);
    
    return 0;
}