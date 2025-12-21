%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// DNA PERSISTENT: The loop is INSIDE the silicon
__global__ void dnaPersistentKernel(int8_t* A, int8_t* B, int8_t* out, int N, int iterations) {
    int idx = threadIdx.x + (threadIdx.y * N) + (threadIdx.z * N * N);
    if (idx < (N * N * N)) {
        int8_t sA = A[idx];
        int8_t sB = B[idx];

        // FORCED UNROLLING: Tells the GPU to process 4 steps at once
        #pragma unroll 4
        for (int i = 0; i < iterations; i++) {
            sA = sA + sB;
            // Branchless Modulo 3: 'if (sA >= 3) sA -= 3' 
            // This compiles to a single 'SEL' (Select) instruction
            sA = (sA >= 3) ? (sA - 3) : sA; 
        }
        out[idx] = sA;
    }
}

// TRADITIONAL: The loop is OUTSIDE on the CPU
__global__ void traditionalKernel(int8_t* A, int8_t* B, int8_t* out, int N) {
    int idx = threadIdx.x + (threadIdx.y * N) + (threadIdx.z * N * N);
    if (idx < (N * N * N)) {
        out[idx] = (A[idx] + B[idx]) % 3;
    }
}

int main() {
    const int N = 6;
    const int total = N * N * N;
    const int iterations = 1000000;
    int8_t *gridA, *gridB, *gridOut;

    cudaMallocManaged(&gridA, total);
    cudaMallocManaged(&gridB, total);
    cudaMallocManaged(&gridOut, total);

    for (int i = 0; i < total; i++) { gridA[i] = 1; gridB[i] = 1; }

    dim3 blockSize(N, N, N);

    // --- 1. TRADITIONAL MILLION ---
    std::cout << "Starting 1,000,000 Traditional Launches..." << std::endl;
    auto startTrad = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        traditionalKernel<<<1, blockSize>>>(gridA, gridB, gridOut, N);
    }
    cudaDeviceSynchronize();
    auto endTrad = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTrad = endTrad - startTrad;

    // --- 2. DNA PERSISTENT MILLION ---
    std::cout << "Starting 1,000,000 DNA Persistent Cycles..." << std::endl;
    auto startDNA = std::chrono::high_resolution_clock::now();
    dnaPersistentKernel<<<1, blockSize>>>(gridA, gridB, gridOut, N, iterations);
    cudaDeviceSynchronize();
    auto endDNA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedDNA = endDNA - startDNA;

    // --- RESULTS ---
    std::cout << "\n--- 21/12 RELEASE: THE MILLION CYCLE GAP ---" << std::endl;
    std::cout << "Traditional (CPU-Driven): " << elapsedTrad.count() << " ms" << std::endl;
    std::cout << "DNA Persistent (Silicon): " << elapsedDNA.count() << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "PERFORMANCE GAP: " << (elapsedTrad.count() / elapsedDNA.count()) << "x Faster" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    cudaFree(gridA); cudaFree(gridB); cudaFree(gridOut);
    return 0;
}