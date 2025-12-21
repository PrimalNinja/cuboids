%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>

// --- THE HYPER-OPTIMIZED DNA INTERFERENCE ---
__global__ void interferenceKernel(uint8_t* __restrict__ A, uint8_t* __restrict__ B, uint8_t* __restrict__ out, int TOTAL, int iterations) {
    // Grid-stride loop for max throughput
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < TOTAL; i += stride) {
        // REGISTER PINNING: Load once, stay in silicon
        uint8_t sA = A[i];
        uint8_t sB = B[i];

        // DNA PERSISTENCE: Zero-Modulo Branchless Logic
        for(int j = 0; j < iterations; j++) {
            sA = sA + sB;
            // Branchless Modulo 3: if (sA >= 3) sA -= 3;
            // Faster than % because it uses 'SEL' or 'LOP3' hardware instructions
            sA = (sA >= 3) ? (sA - 3) : sA;
        }

        out[i] = sA;
    }
}

int main() {
    const int N = 256; 
    const size_t total = (size_t)N * N * N;
    const int iter = 100;
    uint8_t *gridA, *gridB, *gridOut;

    cudaMallocManaged(&gridA, total);
    cudaMallocManaged(&gridB, total);
    cudaMallocManaged(&gridOut, total);

    // Initial Conditions
    for (size_t i = 0; i < total; i++) { gridA[i] = 1; gridB[i] = 0; }
    for (int i = 0; i < N; i++) gridB[i + (i * N) + (i * N * N)] = 1;

    cudaEvent_t start, stop;
    float timeTraditional, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Execution Configuration for Max Occupancy
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    std::cout << "--- 21/12 INTERFERENCE ASCENSION: N=" << N << " ---" << std::endl;

    // 1. TRADITIONAL (100x Memory Pressure)
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++) {
        interferenceKernel<<<blocks, threads>>>(gridA, gridB, gridOut, total, 1);
    }
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeTraditional, start, stop);

    // 2. DNA PERSISTENT (1x Memory Pressure)
    cudaEventRecord(start);
    interferenceKernel<<<blocks, threads>>>(gridA, gridB, gridOut, total, iter);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeDNA, start, stop);

    std::cout << "Traditional (VRAM Bound): " << timeTraditional << " ms" << std::endl;
    std::cout << "DNA (Instruction Bound):  " << timeDNA << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "PERFORMANCE GAP: " << timeTraditional / timeDNA << "x" << std::endl;

    cudaFree(gridA); cudaFree(gridB); cudaFree(gridOut);
    return 0;
}