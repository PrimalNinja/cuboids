%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <random>

#define TEST_ID 32.03
#define N 64           
#define BATCH_SIZE 64  
#define GEN_COUNT 100  
#define MUTATION_RATE 0.02f
#define THREADS_PER_BLOCK 256

// Optimized ternary logic
__device__ __forceinline__ int8_t ternaryLogicFast(int8_t a, int8_t b) {
    return (a == 0 || b == 0) ? 0 : (a == b) ? 1 : -1;
}

// Optimized evaluation kernel - each block processes one brain
__global__ void turboEvolutionKernel(int8_t* target, int8_t* brains, int* energies) {
    extern __shared__ int sharedEnergy[];
    
    int b = blockIdx.x;  // Each block processes one brain
    int tid = threadIdx.x;
    int totalVoxels = N * N * N;
    
    // Initialize shared memory
    if (tid < 32) {
        sharedEnergy[tid] = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple voxels
    int threadSum = 0;
    for (int voxelIdx = tid; voxelIdx < totalVoxels; voxelIdx += blockDim.x) {
        int8_t t = target[voxelIdx];
        int8_t b_gene = brains[b * totalVoxels + voxelIdx];
        threadSum += (int)ternaryLogicFast(t, b_gene);
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        threadSum += __shfl_down_sync(0xFFFFFFFF, threadSum, offset);
    }
    
    // Store warp sum to shared memory
    int warpId = tid / 32;
    int laneId = tid % 32;
    if (laneId == 0) {
        sharedEnergy[warpId] = threadSum;
    }
    __syncthreads();
    
    // Final reduction
    if (tid == 0) {
        int blockSum = 0;
        int warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < warps; w++) {
            blockSum += sharedEnergy[w];
        }
        energies[b] = blockSum;
    }
}

// Initialize RNG states
__global__ void initRNGStates(curandState* states, unsigned long seed1, unsigned long seed2) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed1 ^ tid, seed2, 0, &states[tid]);
}

// Optimized mutation kernel with proper RNG
__global__ void turboMutateKernel(int8_t* brains, int winnerIdx, 
                                 curandState* rngStates, int totalVoxels) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < totalVoxels) {
        // Each thread handles one voxel position across all brains
        curandState localState = rngStates[tid];
        int8_t championGene = brains[winnerIdx * totalVoxels + tid];
        
        for (int b = 0; b < BATCH_SIZE; b++) {
            if (b == winnerIdx) continue;
            
            float randVal = curand_uniform(&localState);
            if (randVal > MUTATION_RATE) {
                brains[b * totalVoxels + tid] = championGene;
            } else {
                float mutVal = curand_uniform(&localState);
                if (mutVal < 0.5f) {
                    brains[b * totalVoxels + tid] = 1;
                } else if (mutVal < 0.75f) {
                    brains[b * totalVoxels + tid] = -1;
                } else {
                    brains[b * totalVoxels + tid] = 0;
                }
            }
        }
        
        rngStates[tid] = localState;
    }
}

// Selection kernel
__global__ void turboSelectionKernel(int* energies, int* winnerIdx, int* bestEnergy) {
    __shared__ int sharedBest[32];
    __shared__ int sharedIdx[32];
    
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    int localBest = -999999;
    int localIdx = 0;
    
    for (int i = laneId; i < BATCH_SIZE; i += 32) {
        if (energies[i] > localBest) {
            localBest = energies[i];
            localIdx = i;
        }
    }
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        int otherBest = __shfl_down_sync(0xFFFFFFFF, localBest, offset);
        int otherIdx = __shfl_down_sync(0xFFFFFFFF, localIdx, offset);
        if (otherBest > localBest) {
            localBest = otherBest;
            localIdx = otherIdx;
        }
    }
    
    if (laneId == 0) {
        sharedBest[warpId] = localBest;
        sharedIdx[warpId] = localIdx;
    }
    __syncthreads();
    
    if (tid == 0) {
        int globalBest = -999999;
        int globalIdx = 0;
        int warps = min(32, (blockDim.x + 31) / 32);
        
        for (int w = 0; w < warps; w++) {
            if (sharedBest[w] > globalBest) {
                globalBest = sharedBest[w];
                globalIdx = sharedIdx[w];
            }
        }
        
        *winnerIdx = globalIdx;
        *bestEnergy = globalBest;
    }
}

int main() {
    std::random_device rd;
    unsigned long baseSeed1 = rd();
    unsigned long baseSeed2 = rd();
    
    std::cout << "Turbo Evolution Seeds: " << baseSeed1 << ", " << baseSeed2 << std::endl;

    size_t totalVoxels = (size_t)N * N * N;
    size_t brainSize = BATCH_SIZE * totalVoxels;
    
    // Allocate Unified Memory
    int8_t *d_target, *d_brains;
    int *d_energies, *d_winner, *d_bestEnergy;
    curandState *d_rngStates;
    
    cudaMallocManaged(&d_target, totalVoxels);
    cudaMallocManaged(&d_brains, brainSize);
    cudaMallocManaged(&d_energies, BATCH_SIZE * sizeof(int));
    cudaMallocManaged(&d_winner, sizeof(int));
    cudaMallocManaged(&d_bestEnergy, sizeof(int));
    cudaMalloc(&d_rngStates, totalVoxels * sizeof(curandState));

    // Initialize target
    for(int i = 0; i < totalVoxels; i++) d_target[i] = 0;
    for(int i = 0; i < 100; i++) d_target[(i * 1337) % totalVoxels] = 1;

    // Initialize brains with quality random
    std::mt19937 gen(baseSeed1);
    std::uniform_int_distribution<> dist(-1, 1);
    for(size_t i = 0; i < brainSize; i++) {
        d_brains[i] = static_cast<int8_t>(dist(gen));
    }

    // Initialize RNG states on device
    dim3 rngBlocks((totalVoxels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    initRNGStates<<<rngBlocks, THREADS_PER_BLOCK>>>(d_rngStates, baseSeed1, baseSeed2);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\nStarting Turbo Evolution [" << TEST_ID << "]" << std::endl;
    std::cout << "Mutation Rate: " << (MUTATION_RATE * 100) << "%" << std::endl;
    std::cout << "Target: 100 active voxels" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    cudaEventRecord(start);

    int progress[6] = {0};
    int progressPoints[6] = {0, 20, 40, 60, 80, 99};

    size_t sharedMemSize = ((THREADS_PER_BLOCK + 31) / 32) * sizeof(int);

    for (int generation = 0; generation < GEN_COUNT; generation++) {
        // Evaluation phase - one block per brain
        turboEvolutionKernel<<<BATCH_SIZE, THREADS_PER_BLOCK, sharedMemSize>>>(
            d_target, d_brains, d_energies);
        
        // Selection
        turboSelectionKernel<<<1, THREADS_PER_BLOCK>>>(d_energies, d_winner, d_bestEnergy);
        cudaDeviceSynchronize();
        
        // Mutation
        turboMutateKernel<<<rngBlocks, THREADS_PER_BLOCK>>>(
            d_brains, *d_winner, d_rngStates, totalVoxels);
        
        // Record progress
        for (int p = 0; p < 6; p++) {
            if (generation == progressPoints[p]) {
                progress[p] = *d_bestEnergy;
                if (p > 0) {
                    std::cout << "Gen " << generation << ": " << *d_bestEnergy 
                              << " (+" << (progress[p] - progress[p-1]) << ")" << std::endl;
                } else {
                    std::cout << "Gen 0: " << *d_bestEnergy << " (baseline)" << std::endl;
                }
            }
        }
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double processed_voxels = (double)totalVoxels * BATCH_SIZE * GEN_COUNT;
    double gvox_per_sec = (processed_voxels / (ms / 1000.0)) / 1e9;

    std::cout << "------------------------------------" << std::endl;
    std::cout << "F1 BENCHMARK - TEST ID: " << TEST_ID << std::endl;
    std::cout << "Total Time:       " << ms << " ms" << std::endl;
    std::cout << "Throughput:       " << gvox_per_sec << " Giga-Voxels/sec" << std::endl;
    std::cout << "Final Best Score: " << *d_bestEnergy << " / 100" << std::endl;
    std::cout << "Learning Curve:   ";
    for (int p = 0; p < 6; p++) {
        std::cout << progress[p];
        if (p < 5) std::cout << " → ";
    }
    std::cout << std::endl;
    
    if (*d_bestEnergy >= 95) {
        std::cout << "STATUS: TURBO EVOLUTION SUCCESS" << std::endl;
    } else if (*d_bestEnergy >= 80) {
        std::cout << "STATUS: GOOD EVOLUTION" << std::endl;
    } else {
        std::cout << "STATUS: NEEDS OPTIMIZATION" << std::endl;
    }
    
    std::cout << "------------------------------------" << std::endl;

    // Cleanup
    cudaFree(d_target);
    cudaFree(d_brains);
    cudaFree(d_energies);
    cudaFree(d_winner);
    cudaFree(d_bestEnergy);
    cudaFree(d_rngStates);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}