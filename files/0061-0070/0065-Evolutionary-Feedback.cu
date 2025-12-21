%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define TEST_ID 28.02
#define N 32
#define BATCH_SIZE 64
#define VOXEL_COUNT (N*N*N)

// Logic Gate: +1 for match, -1 for conflict
__device__ int8_t ternaryLogic(int8_t a, int8_t b) {
    if (a == 0 || b == 0) return 0;
    return (a == b) ? 1 : -1;
}

// INFERENCE: Evaluate all 64 brains against the target
__global__ void evolutionKernel(int8_t* target, int8_t* brains, int* energies) {
    int b = blockIdx.y; 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < VOXEL_COUNT) {
        int8_t match = ternaryLogic(target[tid], brains[b * VOXEL_COUNT + tid]);
        atomicAdd(&energies[b], (int)match);
    }
}

// SELECTION: Find the Alpha Brain
__global__ void selectionKernel(int* energies, int* winnerIdx, int* maxEnergy) {
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
        *maxEnergy = bestVal;
    }
}

int main() {
    int8_t *d_target, *d_brains;
    int *d_energies, *d_winner, *d_max;
    
    cudaMallocManaged(&d_target, VOXEL_COUNT);
    cudaMallocManaged(&d_brains, BATCH_SIZE * VOXEL_COUNT);
    cudaMallocManaged(&d_energies, BATCH_SIZE * sizeof(int));
    cudaMallocManaged(&d_winner, sizeof(int));
    cudaMallocManaged(&d_max, sizeof(int));

    // Define the "Identity" to be matched
    for(int i=0; i<VOXEL_COUNT; i++) d_target[i] = (i % 7 == 0) ? 1 : 0;

    // Initial Random Population
    for(int i=0; i<BATCH_SIZE * VOXEL_COUNT; i++) d_brains[i] = (rand() % 3) - 1;

    std::cout << "--- F1 TELEMETRY [ID: " << TEST_ID << "] ---" << std::endl;

    for (int gen = 0; gen < 5; gen++) {
        for(int i=0; i<BATCH_SIZE; i++) d_energies[i] = 0;

        evolutionKernel<<<dim3(VOXEL_COUNT/256 + 1, BATCH_SIZE), 256>>>(d_target, d_brains, d_energies);
        cudaDeviceSynchronize();

        selectionKernel<<<1, 1>>>(d_energies, d_winner, d_max);
        cudaDeviceSynchronize();

        std::cout << "Gen " << gen << " | Winner: " << *d_winner << " | Energy: " << *d_max << std::endl;
        
        // FUTURE: Insert Mutation/Replication Logic here
    }

    std::cout << "STATUS: MANIFOLD PERSISTENT" << std::endl;
    std::cout << "------------------------------------" << std::endl;

    cudaFree(d_target); cudaFree(d_brains); cudaFree(d_energies); 
    cudaFree(d_winner); cudaFree(d_max);
    return 0;
}