%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>

// Using a struct to force 16-byte alignment (128-bit)
struct Voxel16 {
    uint32_t w[4]; 
};

__global__ void interferenceKernel(Voxel16* __restrict__ A, Voxel16* __restrict__ B, Voxel16* __restrict__ out, int TOTAL, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < TOTAL) {
        // VECTOR LOAD: 128-bits pulled directly into Registers
        Voxel16 regA = A[idx];
        Voxel16 regB = B[idx];

        // DNA PERSISTENCE: 100 cycles of parallel evolution
        for(int j = 0; j < iterations; j++) {
            // Process 4 words (16 voxels) in parallel using SWAR logic
            #pragma unroll
            for(int k = 0; k < 4; k++) {
                uint32_t a = regA.w[k];
                uint32_t b = regB.w[k];
                
                // Parallel Addition
                a += b;
                
                // Parallel Branchless Modulo-3 for 4 voxels inside the uint32
                uint32_t mask = 0;
                if (((a >> 0)  & 0xFF) >= 3) mask |= (3u << 0);
                if (((a >> 8)  & 0xFF) >= 3) mask |= (3u << 8);
                if (((a >> 16) & 0xFF) >= 3) mask |= (3u << 16);
                if (((a >> 24) & 0xFF) >= 3) mask |= (3u << 24);
                
                regA.w[k] = a - mask;
            }
        }
        // VECTOR COMMIT
        out[idx] = regA;
    }
}

int main() {
    const int N = 512; // Forced Memory Wall (134 million voxels)
    const size_t total_voxels = (size_t)N * N * N;
    const size_t total_vectors = total_voxels / 16; 
    const int iter = 100;

    Voxel16 *gridA, *gridB, *gridOut;
    cudaMallocManaged(&gridA, total_voxels);
    cudaMallocManaged(&gridB, total_voxels);
    cudaMallocManaged(&gridOut, total_voxels);

    // Initial Substrate
    uint8_t* pA = (uint8_t*)gridA;
    uint8_t* pB = (uint8_t*)gridB;
    for (size_t i = 0; i < total_voxels; i++) { pA[i] = 1; pB[i] = 1; }

    cudaEvent_t start, stop;
    float timeTraditional, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int threads = 128; // Smaller block for higher occupancy
    int blocks = (total_vectors + threads - 1) / threads;

    std::cout << "--- N=512 VECTORIZED DNA ASCENSION ---" << std::endl;

    // 1. TRADITIONAL
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++) {
        interferenceKernel<<<blocks, threads>>>(gridA, gridB, gridOut, total_vectors, 1);
    }
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeTraditional, start, stop);

    // 2. DNA PERSISTENT
    cudaEventRecord(start);
    interferenceKernel<<<blocks, threads>>>(gridA, gridB, gridOut, total_vectors, iter);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeDNA, start, stop);

    std::cout << "Traditional (VRAM Bound): " << timeTraditional << " ms" << std::endl;
    std::cout << "DNA Vector (Reg Bound):  " << timeDNA << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "FINAL PERFORMANCE GAP: " << timeTraditional / timeDNA << "x" << std::endl;

    cudaFree(gridA); cudaFree(gridB); cudaFree(gridOut);
    return 0;
}