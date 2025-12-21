%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <random>
#include <ctime>

#define N 64
#define BATCH_SIZE 64
#define TOTAL_VOXELS (N * N * N)
#define MUTATION_RATE 0.02f

// --- FUSED KERNEL: INFERENCE + SHARED REDUCTION ---
__global__ void fusedEvolutionKernel(int8_t* target, int8_t* brains, int* energies) {
    __shared__ int blockCache[256];
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    int matchCount = 0;
    if (tid < TOTAL_VOXELS) {
        int8_t t = target[tid];
        int8_t b_gene = brains[b * TOTAL_VOXELS + tid];
        if (t != 0 && b_gene != 0) matchCount = (t == b_gene) ? 1 : -1;
    }

    blockCache[cacheIdx] = matchCount;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (cacheIdx < i) blockCache[cacheIdx] += blockCache[cacheIdx + i];
        __syncthreads();
    }

    if (cacheIdx == 0) atomicAdd(&energies[b], blockCache[0]);
}

// --- MUTATION KERNEL WITH TRUE RANDOMNESS ---
__global__ void mutateKernel(int8_t* brains, int winnerIdx, unsigned long seed1, unsigned long seed2) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < TOTAL_VOXELS) {
        int8_t championGene = brains[winnerIdx * TOTAL_VOXELS + tid];
        
        // UNIQUE seed for each thread AND each generation
        curandState state;
        curand_init(seed1 + tid, seed2, 0, &state);
        
        for (int b = 0; b < BATCH_SIZE; b++) {
            if (b == winnerIdx) continue;
            
            float shouldMutate = curand_uniform(&state);
            
            if (shouldMutate > MUTATION_RATE) {
                brains[b * TOTAL_VOXELS + tid] = championGene;
            } else {
                float mutationValue = curand_uniform(&state);
                if (mutationValue < 0.33f) {
                    brains[b * TOTAL_VOXELS + tid] = 1;
                } else if (mutationValue < 0.66f) {
                    brains[b * TOTAL_VOXELS + tid] = -1;
                } else {
                    brains[b * TOTAL_VOXELS + tid] = 0;
                }
            }
        }
    }
}

// --- SELECTION KERNEL ---
__global__ void selectionKernel(int* energies, int* winnerIdx, int* bestEnergy) {
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
        *bestEnergy = bestVal;
    }
}

int main() {
    // TRUE RANDOM SEEDS - Different each run
    std::random_device rd;
    unsigned long runSeed1 = rd();
    unsigned long runSeed2 = rd();
    
    // Use time as additional entropy
    unsigned long timeSeed = static_cast<unsigned long>(std::chrono::system_clock::now().time_since_epoch().count());
    
    std::cout << "Run Seeds: " << runSeed1 << ", " << runSeed2 
              << ", Time: " << timeSeed << std::endl;

    int8_t *d_target, *d_brains;
    int *d_energies, *d_winner, *d_bestEnergy;
    
    cudaMallocManaged(&d_target, TOTAL_VOXELS);
    cudaMallocManaged(&d_brains, BATCH_SIZE * TOTAL_VOXELS);
    cudaMallocManaged(&d_energies, BATCH_SIZE * sizeof(int));
    cudaMallocManaged(&d_winner, sizeof(int));
    cudaMallocManaged(&d_bestEnergy, sizeof(int));

    // TARGET SETUP
    for(int i = 0; i < TOTAL_VOXELS; i++) d_target[i] = 0;
    for(int i = 0; i < 100; i++) d_target[(i * 997) % TOTAL_VOXELS] = 1;
    
    // TRULY RANDOM INITIAL POPULATION
    std::mt19937 gen(runSeed1 ^ timeSeed);
    std::uniform_int_distribution<> dist(-1, 1);
    
    for(int i = 0; i < BATCH_SIZE * TOTAL_VOXELS; i++) {
        d_brains[i] = static_cast<int8_t>(dist(gen));
    }

    // Verify initial diversity
    int diverse = 0;
    for(int i = 1; i < BATCH_SIZE * TOTAL_VOXELS; i++) {
        if(d_brains[i] != d_brains[0]) diverse++;
    }
    std::cout << "Initial population diversity: " 
              << (100.0 * diverse / (BATCH_SIZE * TOTAL_VOXELS)) << "%" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    std::cout << "\n------------------------------------" << std::endl;
    std::cout << "F1 TELEMETRY - TEST ID: 34.03 | TRUE RANDOM EVOLUTION" << std::endl;
    std::cout << "Mutation Rate: " << (MUTATION_RATE * 100) << "%" << std::endl;
    std::cout << "Target Pattern: 100 active voxels" << std::endl;
    
    cudaEventRecord(start);
    
    // Track learning curve for comparison across runs
    int learningCurve[6] = {0};
    
    for(int generation = 0; generation < 50; generation++) {
        cudaMemset(d_energies, 0, BATCH_SIZE * sizeof(int));
        
        dim3 evalBlocks(TOTAL_VOXELS/256 + 1, BATCH_SIZE);
        fusedEvolutionKernel<<<evalBlocks, 256>>>(d_target, d_brains, d_energies);
        cudaDeviceSynchronize();
        
        selectionKernel<<<1, 1>>>(d_energies, d_winner, d_bestEnergy);
        cudaDeviceSynchronize();
        
        // DIFFERENT seed each generation AND each run
        unsigned long seed1 = runSeed1 + generation * 7919;
        unsigned long seed2 = runSeed2 + generation * 104729;
        
        dim3 mutateBlocks(TOTAL_VOXELS/256 + 1, 1);
        mutateKernel<<<mutateBlocks, 256>>>(d_brains, *d_winner, seed1, seed2);
        cudaDeviceSynchronize();
        
        // Record learning curve at key points
        if (generation == 0) learningCurve[0] = *d_bestEnergy;
        if (generation == 10) learningCurve[1] = *d_bestEnergy;
        if (generation == 20) learningCurve[2] = *d_bestEnergy;
        if (generation == 30) learningCurve[3] = *d_bestEnergy;
        if (generation == 40) learningCurve[4] = *d_bestEnergy;
        if (generation == 49) learningCurve[5] = *d_bestEnergy;
    }
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    double gvox = (double)TOTAL_VOXELS * BATCH_SIZE * 50 / (ms/1000.0) / 1e9;
    
    std::cout << "\n--- RESULTS (THIS RUN) ---" << std::endl;
    std::cout << "Learning Curve: ";
    std::cout << "Gen 0=" << learningCurve[0] << ", ";
    std::cout << "10=" << learningCurve[1] << ", ";
    std::cout << "20=" << learningCurve[2] << ", ";
    std::cout << "30=" << learningCurve[3] << ", ";
    std::cout << "40=" << learningCurve[4] << ", ";
    std::cout << "49=" << learningCurve[5] << std::endl;
    
    std::cout << "Total Time: " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << gvox << " Giga-Voxels/sec" << std::endl;
    std::cout << "Final Best Score: " << *d_bestEnergy << " / 100" << std::endl;
    
    // Performance assessment
    if (*d_bestEnergy >= 95) {
        std::cout << "STATUS: EXCELLENT EVOLUTION" << std::endl;
    } else if (*d_bestEnergy >= 80) {
        std::cout << "STATUS: GOOD EVOLUTION" << std::endl;
    } else if (*d_bestEnergy >= 60) {
        std::cout << "STATUS: MODERATE EVOLUTION" << std::endl;
    } else {
        std::cout << "STATUS: POOR EVOLUTION" << std::endl;
    }
    
    std::cout << "\nNOTE: Run 3 times to verify evolutionary variance" << std::endl;
    std::cout << "      (Should see different learning curves)" << std::endl;
    std::cout << "------------------------------------\n" << std::endl;

    cudaFree(d_target); 
    cudaFree(d_brains); 
    cudaFree(d_energies); 
    cudaFree(d_winner);
    cudaFree(d_bestEnergy);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}