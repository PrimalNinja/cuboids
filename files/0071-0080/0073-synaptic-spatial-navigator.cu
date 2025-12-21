%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

#define N 64
#define BATCH_SIZE 64
#define TOTAL_VOXELS (N * N * N)
#define GEN_COUNT 50

struct SpatialDNA {
    float tx, ty, tz;
};

// --- DNA KERNEL: OPTIMIZED SPATIAL INFERENCE ---
__global__ void dnaSpatialKernel(int8_t* target, int8_t* brains, SpatialDNA* spatial, int* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    // Pull translation once per block to save registers
    __shared__ int offX, offY, offZ;
    if (threadIdx.x == 0) {
        offX = (int)spatial[b].tx;
        offY = (int)spatial[b].ty;
        offZ = (int)spatial[b].tz;
    }
    __syncthreads();

    int lx = tid % N;
    int ly = (tid / N) % N;
    int lz = tid / (N * N);

    int gx = lx + offX;
    int gy = ly + offY;
    int gz = lz + offZ;

    int match = 0;
    if (gx >= 0 && gx < N && gy >= 0 && gy < N && gz >= 0 && gz < N) {
        // Core DNA logic: The "45x Pulse"
        match = (target[gx + gy*N + gz*N*N] == brains[b * TOTAL_VOXELS + tid]) ? 1 : -1;
    }

    __shared__ int cache[256];
    cache[threadIdx.x] = match;
    __syncthreads();
    for (int i = 128; i > 0; i >>= 1) {
        if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&energies[b], cache[0]);
}

// --- GPU SELECTION: ELIMINATE CPU STALL ---
__global__ void selectionKernel(int* energies, int* winnerIdx) {
    if (threadIdx.x == 0) {
        int bestVal = -1e9;
        int bestIdx = 0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            if (energies[i] > bestVal) {
                bestVal = energies[i];
                bestIdx = i;
            }
        }
        *winnerIdx = bestIdx;
    }
}

__global__ void spatialMutateKernel(int8_t* brains, SpatialDNA* spatial, int* winnerIdx, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    int champIdx = *winnerIdx;
    int8_t champGene = brains[champIdx * TOTAL_VOXELS + tid];
    
    curandState state;
    curand_init(seed, tid, 0, &state);

    for (int b = 0; b < BATCH_SIZE; b++) {
        if (b == champIdx) continue;
        
        // Voxel Mutation
        if (curand_uniform(&state) < 0.05f) {
            brains[b * TOTAL_VOXELS + tid] = (int8_t)((curand_uniform(&state) * 3) - 1);
        } else {
            brains[b * TOTAL_VOXELS + tid] = champGene;
        }

        // Morphological Mutation (only 1 thread handles this per brain)
        if (tid == 0) {
            spatial[b].tx = spatial[champIdx].tx + (curand_uniform(&state) - 0.5f) * 2.0f;
            spatial[b].ty = spatial[champIdx].ty + (curand_uniform(&state) - 0.5f) * 2.0f;
            spatial[b].tz = spatial[champIdx].tz + (curand_uniform(&state) - 0.5f) * 2.0f;
        }
    }
}

int main() {
    int8_t *d_tDNA, *d_bDNA; int *d_eDNA, *d_win; SpatialDNA *d_spatial;
    cudaMallocManaged(&d_tDNA, TOTAL_VOXELS);
    cudaMallocManaged(&d_bDNA, BATCH_SIZE * TOTAL_VOXELS);
    cudaMallocManaged(&d_eDNA, BATCH_SIZE * sizeof(int));
    cudaMallocManaged(&d_win, sizeof(int));
    cudaMallocManaged(&d_spatial, BATCH_SIZE * sizeof(SpatialDNA));

    cudaMemset(d_tDNA, 1, TOTAL_VOXELS);
    cudaMemset(d_bDNA, 0, BATCH_SIZE * TOTAL_VOXELS);
    cudaMemset(d_spatial, 0, BATCH_SIZE * sizeof(SpatialDNA));

    cudaEvent_t s1, e1;
    cudaEventCreate(&s1); cudaEventCreate(&e1);

    std::cout << "--- 0073 SPATIAL NAVIGATION (STABILIZED) ---" << std::endl;

    cudaEventRecord(s1);
    for (int g = 0; g < GEN_COUNT; g++) {
        cudaMemset(d_eDNA, 0, BATCH_SIZE * sizeof(int));
        dnaSpatialKernel<<<dim3(TOTAL_VOXELS/256+1, BATCH_SIZE), 256>>>(d_tDNA, d_bDNA, d_spatial, d_eDNA);
        selectionKernel<<<1, 1>>>(d_eDNA, d_win);
        spatialMutateKernel<<<TOTAL_VOXELS/256+1, 256>>>(d_bDNA, d_spatial, d_win, 1234 + g);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms;
    cudaEventElapsedTime(&ms, s1, e1);
    std::cout << "DNA Spatial Time: " << ms << " ms" << std::endl;
    std::cout << "Final Throughput: " << (double)TOTAL_VOXELS * BATCH_SIZE * GEN_COUNT / (ms/1000.0) / 1e9 << " GVox/s" << std::endl;

    return 0;
}