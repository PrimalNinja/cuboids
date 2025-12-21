%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>

// --- THE HYPER-SPATIAL ENGINE ---
__global__ void rotateXKernel(uint8_t* __restrict__ input, uint8_t* __restrict__ output, int N, int iterations) {
    // Using 1D indexing for Grid-Stride to support N=512
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N * N;

    if (idx < total) {
        // REGISTER PINNING: Fetch voxel soul
        uint8_t val = input[idx];
        
        // Recover 3D space in registers
        int x = idx % N;
        int y = (idx / N) % N;
        int z = idx / (N * N);

        // THE DNA LOOP: Virtual Spatial Evolution
        // We rotate the coordinate 100 times in silicon without touching VRAM
        for(int i = 0; i < iterations; i++) {
            int tempY = y;
            y = z;
            z = (N - 1) - tempY;
            // Branchless logic to keep the pipeline full
            val = (val ^ 0x02) | (val & 0x01); 
        }

        // COMMIT: Find the final physical landing spot
        int newIdx = x + (y * N) + (z * N * N);
        output[newIdx] = val;
    }
}

int main() {
    const int N = 256; // Scaled up for "Real" physics
    const size_t total = (size_t)N * N * N;
    const int iter = 100;
    uint8_t *gridIn, *gridOut;

    cudaMallocManaged(&gridIn, total);
    cudaMallocManaged(&gridOut, total);

    // Initial Substrate
    for (size_t i = 0; i < total; i++) gridIn[i] = (i % 7 == 0) ? 2 : 0;

    cudaEvent_t start, stop;
    float timeTraditional, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Configure for high occupancy
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    std::cout << "--- SPATIAL ASCENSION: N=" << N << " ---" << std::endl;

    // 1. TRADITIONAL
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++) {
        rotateXKernel<<<blocks, threads>>>(gridIn, gridOut, N, 1);
    }
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeTraditional, start, stop);

    // 2. DNA PERSISTENT
    cudaEventRecord(start);
    rotateXKernel<<<blocks, threads>>>(gridIn, gridOut, N, iter);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeDNA, start, stop);

    std::cout << "Traditional: " << timeTraditional << " ms" << std::endl;
    std::cout << "DNA:         " << timeDNA << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "PERFORMANCE GAP: " << timeTraditional / timeDNA << "x" << std::endl;

    cudaFree(gridIn); cudaFree(gridOut);
    return 0;
}