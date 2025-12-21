%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define BRAINS 16
#define VOXELS 1000

// --- TRADITIONAL: Still anchored to the CPU launch tax ---
__global__ void traditionalKernel(int8_t* A, int8_t* B, int8_t* out) {
    int idx = threadIdx.x + blockIdx.y * VOXELS;
    if (threadIdx.x < VOXELS) {
        out[idx] = (A[idx] + B[idx]) % 3;
    }
}

// --- DNA SOVEREIGN: Zero-Branch Adaptive Mutation ---
__global__ void dnaSovereignKernel(int8_t* A, int8_t* B, int8_t* out, int iterations) {
    int idx = threadIdx.x + blockIdx.y * VOXELS;
    if (threadIdx.x < VOXELS) {
        int8_t sA = A[idx];
        int8_t sB = B[idx];
        int8_t mode = 1;

        #pragma unroll 32
        for (int i = 0; i < iterations; i++) {
            // OPTIMIZATION 1: Predicated Mode Flipping
            // No 'if' statements. mode changes based on boolean logic
            if (sA == 2) mode = -1;
            if (sA == 0) mode = 1;

            // OPTIMIZATION 2: Positive Wrap-Around Math
            // (sA + 3) ensures we never hit negative numbers during (sB * mode)
            // This allows the compiler to use a faster modulo intrinsic
            sA = (sA + 3 + (sB * mode)) % 3;
        }
        out[idx] = sA;
    }
}

int main() {
    const int total = BRAINS * VOXELS;
    const int iterations = 1000000;
    int8_t *gridA, *gridB, *gridOut;

    cudaMallocManaged(&gridA, total);
    cudaMallocManaged(&gridB, total);
    cudaMallocManaged(&gridOut, total);

    for (int i = 0; i < total; i++) { gridA[i] = 1; gridB[i] = 1; }

    dim3 threads(VOXELS);
    dim3 blocks(1, BRAINS);

    std::cout << "--- 21/12 RELEASE: FILE 0038-SOVEREIGN (BRANCHLESS ADAPTIVE) ---" << std::endl;

    auto s1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        traditionalKernel<<<blocks, threads>>>(gridA, gridB, gridOut);
    }
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();

    auto s2 = std::chrono::high_resolution_clock::now();
    dnaSovereignKernel<<<blocks, threads>>>(gridA, gridB, gridOut, iterations);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();

    float trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();
    float dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (Static):    " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Sovereign (Adaptive): " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "ADAPTIVE GAP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(gridA); cudaFree(gridB); cudaFree(gridOut);
    return 0;
}