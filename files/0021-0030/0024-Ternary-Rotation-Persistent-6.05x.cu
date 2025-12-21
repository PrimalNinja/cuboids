%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>

// --- THE HYPER-OPTIMIZED DNA ENGINE ---
__global__ void ternaryVoxelKernel(uint8_t* __restrict__ grid, int TOTAL, int iterations) {
    // Grid-stride loop: Ensures 100% occupancy regardless of N
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < TOTAL; i += stride) {
        // FETCH: Single VRAM read into high-speed GPR (Register)
        uint8_t state = grid[i];

        // PERSISTENCE: 100 cycles of logic at ~1.5GHz
        // Optimization: Removing '%' (division) and 'if' (branching)
        for(int j = 0; j < iterations; j++) {
            // Branchless Ternary: (state + 1) if < 2, else 0
            // This maps to a single 'LOP3' or 'SEL' instruction on hardware
            state = (state + 1) & ((state + 1 < 3) ? 0xFF : 0x00);
        }

        // COMMIT: Single VRAM write back to substrate
        grid[i] = state;
    }
}

int main() {
    // Scale to N=512 to reveal the "Memory Wall"
    const int N = 512;
    const size_t total = (size_t)N * N * N; 
    const int iter = 100;
    uint8_t *grid;

    // Allocate Substrate
    cudaMallocManaged(&grid, total);
    for (size_t i = 0; i < total; i++) grid[i] = 1;

    cudaEvent_t start, stop;
    float timeTraditional, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Dynamic Execution Configuration
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // Stay within hardware scheduler limits

    std::cout << "--- 21/12 HYPER-OPTIMIZED ENGINE (N=" << N << ") ---" << std::endl;

    // 1. TRADITIONAL: 100x VRAM Saturation
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++) {
        ternaryVoxelKernel<<<blocks, threads>>>(grid, total, 1);
    }
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeTraditional, start, stop);

    // Reset Substrate
    for (size_t i = 0; i < total; i++) grid[i] = 1;

    // 2. DNA PERSISTENT: Single Stream Execution
    cudaEventRecord(start);
    ternaryVoxelKernel<<<blocks, threads>>>(grid, total, iter);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeDNA, start, stop);

    std::cout << "Traditional (100x Bus Traffic): " << timeTraditional << " ms" << std::endl;
    std::cout << "DNA Persistent (1x Bus Traffic):  " << timeDNA << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "SCALED PERFORMANCE GAP: " << timeTraditional / timeDNA << "x" << std::endl;

    // Logic Check (1 + 100) % 3 = 2
    if (grid[0] == 2) std::cout << "LOGIC STATUS: [OK]" << std::endl;

    cudaFree(grid);
    return 0;
}