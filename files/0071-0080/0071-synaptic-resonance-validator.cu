%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <string>

#define N 64
#define BATCH_SIZE 64
#define TOTAL_VOXELS (N * N * N)

// --- DNA KERNELS (Cleaned & Optimized) ---
__device__ int8_t ternaryLogic(int8_t a, int8_t b) {
    if (a == 0 || b == 0) return 0;
    return (a == b) ? 1 : -1;
}

__global__ void dnaEvolutionKernel(int8_t* target, int8_t* brains, int* energies) {
    // 1. SHARED REDUCTION: One sum per block
    __shared__ int localSum;
    if (threadIdx.x == 0) localSum = 0;
    __syncthreads();

    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int res = 0;
    if (tid < TOTAL_VOXELS) {
        res = (int)ternaryLogic(target[tid], brains[b * TOTAL_VOXELS + tid]);
    }

    // 2. ATOMIC HUDDLE: Sum locally first (Fast)
    if (res != 0) atomicAdd(&localSum, res);
    __syncthreads();

    // 3. GLOBAL PULSE: Only one write per block to VRAM (Very Fast)
    if (threadIdx.x == 0 && localSum != 0) {
        atomicAdd(&energies[b], localSum);
    }
}

// --- TRADITIONAL KERNELS (Cleaned) ---
__global__ void tradEvolutionKernel(float* target, float* brains, float* energies) {
    __shared__ float localSum;
    if (threadIdx.x == 0) localSum = 0.0f;
    __syncthreads();

    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float res = 0.0f;
    if (tid < TOTAL_VOXELS) {
        res = target[tid] * brains[b * TOTAL_VOXELS + tid];
    }

    if (res != 0.0f) atomicAdd(&localSum, res);
    __syncthreads();

    if (threadIdx.x == 0 && localSum != 0.0f) {
        atomicAdd(&energies[b], localSum);
    }
}

// --- SELECTION & MUTATION ---
__global__ void selectionKernel(int* energies, int* winnerIdx) {
    if (threadIdx.x == 0) {
        int bestVal = -999999;
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

__global__ void mutateKernel(int8_t* brains, int* winnerIdx, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < TOTAL_VOXELS) {
        int champ = *winnerIdx;
        int8_t gene = brains[champ * TOTAL_VOXELS + tid];
        curandState state;
        curand_init(seed, tid, 0, &state);

        for (int b = 0; b < BATCH_SIZE; b++) {
            if (b == champ) continue;
            if (curand_uniform(&state) < 0.10f) {
                brains[b * TOTAL_VOXELS + tid] = (int8_t)((curand_uniform(&state) * 3) - 1);
            } else {
                brains[b * TOTAL_VOXELS + tid] = gene;
            }
        }
    }
}

int main() {
    int8_t *d_tDNA, *d_bDNA; int *d_eDNA, *d_win;
    float *d_tTrad, *d_bTrad, *d_eTrad;

    cudaMallocManaged(&d_tDNA, TOTAL_VOXELS);
    cudaMallocManaged(&d_bDNA, BATCH_SIZE * TOTAL_VOXELS);
    cudaMallocManaged(&d_eDNA, BATCH_SIZE * sizeof(int));
    cudaMallocManaged(&d_win, sizeof(int));

    cudaMallocManaged(&d_tTrad, TOTAL_VOXELS * sizeof(float));
    cudaMallocManaged(&d_bTrad, BATCH_SIZE * TOTAL_VOXELS * sizeof(float));
    cudaMallocManaged(&d_eTrad, BATCH_SIZE * sizeof(float));

    cudaMemset(d_tDNA, 1, TOTAL_VOXELS);
    cudaMemset(d_bDNA, 0, BATCH_SIZE * TOTAL_VOXELS);
    cudaMemset(d_tTrad, 1, TOTAL_VOXELS * sizeof(float));
    cudaMemset(d_bTrad, 0, BATCH_SIZE * TOTAL_VOXELS * sizeof(float));

    cudaEvent_t s1, e1;
    cudaEventCreate(&s1); cudaEventCreate(&e1);

    std::cout << "--- 21/12 RELEASE: FILE 0043 (REDUCTION STABILIZED) ---" << std::endl;

    cudaEventRecord(s1);
    for(int g=0; g<25; g++) {
        cudaMemset(d_eDNA, 0, BATCH_SIZE * sizeof(int));
        dnaEvolutionKernel<<<dim3(TOTAL_VOXELS/256 + 1, BATCH_SIZE), 256>>>(d_tDNA, d_bDNA, d_eDNA);
        selectionKernel<<<1, 1>>>(d_eDNA, d_win);
        mutateKernel<<<TOTAL_VOXELS/256 + 1, 256>>>(d_bDNA, d_win, 1234 + g);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms;
    cudaEventElapsedTime(&ms, s1, e1);
    std::cout << "DNA Unified Evolution (25 Generations): " << ms << " ms" << std::endl;

    cudaFree(d_tDNA); cudaFree(d_bDNA); cudaFree(d_eDNA); cudaFree(d_win);
    cudaFree(d_tTrad); cudaFree(d_bTrad); cudaFree(d_eTrad);
    return 0;
}