%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1048576 // 1 Million Voxel Comparison

// --- DNA PERSISTENT: Ternary Identity Logic ---
__device__ int8_t ternaryLogic(int8_t world, int8_t memory) {
    if (world == 0 || memory == 0) return 0;
    return (world == memory) ? 1 : -1;
}

__global__ void dnaInference(int8_t* world, int8_t* memory, int* energy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int8_t res = ternaryLogic(world[idx], memory[idx]);
        if (res != 0) atomicAdd(energy, (int)res);
    }
}

// --- TRADITIONAL: FP32 Multiply-Accumulate ---
__global__ void tradInference(float* world, float* memory, float* energy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(energy, world[idx] * memory[idx]);
    }
}

int main() {
    int8_t *d_wT, *d_mT; int *d_eT;
    float *d_wF, *d_mF, *d_eF;

    cudaMallocManaged(&d_wT, N); cudaMallocManaged(&d_mT, N); cudaMallocManaged(&d_eT, sizeof(int));
    cudaMallocManaged(&d_wF, N*4); cudaMallocManaged(&d_mF, N*4); cudaMallocManaged(&d_eF, sizeof(float));

    std::cout << "--- 21/12 RELEASE: FILE 0062 (INFERENCE RACE) ---" << std::endl;

    // Benchmark DNA
    auto s1 = std::chrono::high_resolution_clock::now();
    dnaInference<<< (N+255)/256, 256 >>>(d_wT, d_mT, d_eT);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    
    // Benchmark Traditional
    auto s2 = std::chrono::high_resolution_clock::now();
    tradInference<<< (N+255)/256, 256 >>>(d_wF, d_mF, d_eF);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();

    double dna_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();
    double trad_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Ternary Logic Latency: " << dna_ms << " ms" << std::endl;
    std::cout << "Standard FP32 Latency: " << trad_ms << " ms" << std::endl;
    std::cout << "EFFICIENCY RATIO:      " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_wT); cudaFree(d_mT); cudaFree(d_eT);
    cudaFree(d_wF); cudaFree(d_mF); cudaFree(d_eF);
    return 0;
}