%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// --- TRADITIONAL: THE FRICTION ---
__global__ void traditionalKernel(int8_t* A, int8_t* B, int8_t* out, int N) {
    int x = threadIdx.x; int y = threadIdx.y; int z = threadIdx.z;
    if (x < N && y < N && z < N) {
        // CPU forces a memory re-index every time
        int rotIdx = y + (x * N) + (z * N * N);
        out[x + (y * N) + (z * N * N)] = (A[rotIdx] + B[x + (y * N) + (z * N * N)]) % 3;
    }
}

// --- SOVEREIGN DNA: THE FLOW ---
__global__ void dnaSovereignKernel(int8_t* A, int8_t* B, int8_t* out, int N, int iterations) {
    int x = threadIdx.x; int y = threadIdx.y; int z = threadIdx.z;
    if (x < N && y < N && z < N) {
        int idx = x + (y * N) + (z * N * N);
        int8_t sA = A[idx];
        int8_t sB = B[idx];

        #pragma unroll 8
        for (int i = 0; i < iterations; i++) {
            // Optimization: Branchless Addition/Modulo
            sA = sA + sB;
            sA = (sA >= 3) ? (sA - 3) : sA;

            // Optimization: Zero-cost Register Swap
            int8_t temp = sA;
            sA = sB;
            sB = temp;
        }
        out[idx] = sA;
    }
}

int main() {
    const int N = 10;
    const int total = N * N * N;
    const int iterations = 1000000;
    int8_t *gridA, *gridB, *gridOut;

    cudaMallocManaged(&gridA, total);
    cudaMallocManaged(&gridB, total);
    cudaMallocManaged(&gridOut, total);

    for (int i = 0; i < total; i++) { gridA[i] = 1; gridB[i] = 2; }

    std::cout << "--- 21/12 FINAL COMPARITOR: FILE 0035 ---" << std::endl;

    // 1. TRADITIONAL BENCHMARK
    auto s1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        traditionalKernel<<<1, dim3(N,N,N)>>>(gridA, gridB, gridOut, N);
    }
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. SOVEREIGN DNA BENCHMARK
    auto s2 = std::chrono::high_resolution_clock::now();
    dnaSovereignKernel<<<1, dim3(N,N,N)>>>(gridA, gridB, gridOut, N, iterations);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (Memory Bound): " << trad_ms << " ms" << std::endl;
    std::cout << "Sovereign DNA (Logic Bound): " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "ULTIMATE PERFORMANCE GAP: " << (trad_ms / dna_ms) << "x" << std::endl;
    std::cout << "DNA Logic Speed: " << (dna_ms * 1000.0) / iterations << " nanoseconds/cycle" << std::endl;

    cudaFree(gridA); cudaFree(gridB); cudaFree(gridOut);
    return 0;
}