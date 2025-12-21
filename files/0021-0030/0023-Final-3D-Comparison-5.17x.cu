%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>

// --- HYPER-OPTIMIZED ENGINE ---
__global__ void rotateTernaryKernel(uint8_t* __restrict__ grid, int total, int iterations) {
    // Grid-stride loop: One thread handles multiple voxels if volume grows
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // LOAD: Pull from VRAM into a High-Speed Register
        uint8_t state = grid[i];

        // DNA PERSISTENT LOOP
        // Use Branchless logic to avoid heavy Integer Modulo (%)
        for(int j = 0; j < iterations; j++) {
            state++;
            if (state == 3) state = 0; 
        }

        // STORE: Write the final "evolved" state back once
        grid[i] = state;
    }
}

int main() {
    // Scaling to N=128 to show the real DNA advantage
    const int N = 128; 
    const int total = N * N * N;
    const int iter = 100;
    uint8_t *d_grid;

    cudaMallocManaged(&d_grid, total);

    // Initial State: 1
    for (int i = 0; i < total; i++) d_grid[i] = 1;

    cudaEvent_t start, stop;
    float timeTraditional, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Configuration for Scaling
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 65535) blocksPerGrid = 65535; // Hardware cap

    std::cout << "--- 21/12 HYPER-OPTIMIZED: N=" << N << " ---" << std::endl;

    // 1. TRADITIONAL METHOD (100 Discrete Launches)
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++) {
        rotateTernaryKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid, total, 1);
    }
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeTraditional, start, stop);

    // Reset
    for (int i = 0; i < total; i++) d_grid[i] = 1;

    // 2. DNA PERSISTENT METHOD (1 Launch, 100 Internal Cycles)
    cudaEventRecord(start);
    rotateTernaryKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid, total, iter);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeDNA, start, stop);

    std::cout << "Traditional (100x Launch): " << timeTraditional << " ms" << std::endl;
    std::cout << "DNA Persistent (1x Launch):  " << timeDNA << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "PERFORMANCE GAP: " << timeTraditional / timeDNA << "x" << std::endl;

    // Final Verification (1 + 100) % 3 = 2
    if (d_grid[0] == 2) std::cout << "LOGIC STATUS: [OK]" << std::endl;

    cudaFree(d_grid);
    return 0;
}