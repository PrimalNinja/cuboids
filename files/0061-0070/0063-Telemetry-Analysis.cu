%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define TEST_ID 26.02
#define N 1048576 // 1M Voxel Manifold

__device__ int8_t ternaryLogic(int8_t world, int8_t memory) {
    if (world == 0 || memory == 0) return 0;
    return (world == memory) ? 1 : -1;
}

__global__ void f1Kernel(int8_t* world, int8_t* memory, int* energy, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int8_t res = ternaryLogic(world[idx], memory[idx]);
        if (res != 0) atomicAdd(energy, (int)res);
    }
}

int main() {
    int8_t *d_world, *d_memory;
    int *d_energy;

    cudaMallocManaged(&d_world, N);
    cudaMallocManaged(&d_memory, N);
    cudaMallocManaged(&d_energy, sizeof(int));

    // SCENARIO: 1M Voxels in Memory
    for(int i=0; i<N; i++) d_memory[i] = 1; 

    // TEST A: PERFECT SYNC (World = Memory)
    for(int i=0; i<N; i++) d_world[i] = 1;
    *d_energy = 0;
    
    auto s1 = std::chrono::high_resolution_clock::now();
    f1Kernel<<<(N+255)/256, 256>>>(d_world, d_memory, d_energy, N);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();

    int resultA = *d_energy;

    // TEST B: ENTROPY (World has 10% Conflict, 20% Noise/Zero)
    for(int i=0; i<N; i++) {
        if (i % 10 == 0) d_world[i] = -1;      // Conflict
        else if (i % 5 == 0) d_world[i] = 0;   // Noise
        else d_world[i] = 1;                  // Match
    }
    *d_energy = 0;
    f1Kernel<<<(N+255)/256, 256>>>(d_world, d_memory, d_energy, N);
    cudaDeviceSynchronize();

    int resultB = *d_energy;

    // OUTPUT
    std::cout << "--- F1 TELEMETRY REPORT [ID: " << TEST_ID << "] ---" << std::endl;
    std::cout << "Manifold Size: " << N << " voxels" << std::endl;
    std::cout << "Baseline Sync: " << resultA << " (Peak Resonance)" << std::endl;
    std::cout << "Noisy Signal:  " << resultB << " (Signal Stability)" << std::endl;
    std::cout << "Processing:    " << std::chrono::duration<double, std::milli>(e1 - s1).count() << " ms" << std::endl;
    
    if (resultB > 0) std::cout << "STATUS: SIGNAL RECOVERED" << std::endl;
    else std::cout << "STATUS: SIGNAL LOST IN CONFLICT" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    cudaFree(d_world); cudaFree(d_memory); cudaFree(d_energy);
    return 0;
}