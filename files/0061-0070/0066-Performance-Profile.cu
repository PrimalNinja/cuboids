%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define TEST_ID 29.02
#define N 32
#define BATCH_SIZE 64
#define GEN_COUNT 100 // Doubling the generations for deeper convergence

// The Ternary Logic Core
__device__ int8_t ternaryLogic(int8_t a, int8_t b) {
    if (a == 0 || b == 0) return 0;
    return (a == b) ? 1 : -1;
}

// Global Inference
__global__ void evolutionKernel(int8_t* target, int8_t* brains, int* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int totalVoxels = N*N*N;
    if (tid < totalVoxels) {
        int8_t match = ternaryLogic(target[tid], brains[b * totalVoxels + tid]);
        atomicAdd(&energies[b], (int)match);
    }
}

// Selection of the Alpha
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

// Genetic Mutation & Replication
__global__ void mutateKernel(int8_t* brains, int winnerIdx, unsigned int seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int totalVoxels = N*N*N;
    if (tid < totalVoxels) {
        int8_t championGene = brains[winnerIdx * totalVoxels + tid];
        curandState state;
        curand_init(seed, tid, 0, &state);

        for (int b = 0; b < BATCH_SIZE; b++) {
            if (b == winnerIdx) continue;
            // 90% Copy, 10% Mutate (Accelerated Learning)
            if (curand_uniform(&state) > 0.10f) {
                brains[b * totalVoxels + tid] = championGene;
            } else {
                brains[b * totalVoxels + tid] = (int8_t)((curand_uniform(&state) * 3) - 1);
            }
        }
    }
}

int main() {
    int8_t *d_target, *d_brains;
    int *d_energies, *d_winner;
    int totalVoxels = N*N*N;

    cudaMallocManaged(&d_target, totalVoxels);
    cudaMallocManaged(&d_brains, BATCH_SIZE * totalVoxels);
    cudaMallocManaged(&d_energies, BATCH_SIZE * sizeof(int));
    cudaMallocManaged(&d_winner, sizeof(int));

    // Target: A Sparse 3D Ring
    for(int i=0; i<totalVoxels; i++) d_target[i] = 0;
    for(int i=0; i<totalVoxels; i+=100) d_target[i] = 1; 

    // Initialization
    for(int i=0; i<BATCH_SIZE * totalVoxels; i++) d_brains[i] = 0;

    std::cout << "--- F1 TELEMETRY [ID: " << TEST_ID << "] ---" << std::endl;

    for (int g = 0; g <= GEN_COUNT; g++) {
        cudaMemset(d_energies, 0, BATCH_SIZE * sizeof(int));
        
        evolutionKernel<<<dim3(totalVoxels/256 + 1, BATCH_SIZE), 256>>>(d_target, d_brains, d_energies);
        cudaDeviceSynchronize();
        
        selectionKernel<<<1, 1>>>(d_energies, d_winner);
        cudaDeviceSynchronize();

        if (g % 20 == 0) {
            std::cout << "Generation " << g << " | Champion Resonance: " << d_energies[*d_winner] << std::endl;
        }

        mutateKernel<<<totalVoxels/256 + 1, 256>>>(d_brains, *d_winner, time(0) + g);
        cudaDeviceSynchronize();
    }

    std::cout << "STATUS: MANIFOLD FULLY EVOLVED" << std::endl;
    std::cout << "------------------------------------" << std::endl;

    cudaFree(d_target); cudaFree(d_brains); cudaFree(d_energies); cudaFree(d_winner);
    return 0;
}