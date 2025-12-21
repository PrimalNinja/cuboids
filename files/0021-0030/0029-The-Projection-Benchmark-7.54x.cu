%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void collapseKernel(uint32_t* __restrict__ volume, int* __restrict__ floor, int N, int iterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && z < N) {
        int floorIdx = x + (z * N);
        int finalSum = 0;

        // DNA PERSISTENCE: Loop unrolling helps the ALU pipeline math
        #pragma unroll 4
        for(int i = 0; i < iterations; i++) {
            int currentSum = 0;
            // STRIDED Y-AXIS: Process 4 voxels per iteration
            #pragma unroll 8
            for (int y = 0; y < N / 4; y++) {
                uint32_t chunk = volume[(x + (y * 4 * N) + (z * N * N)) / 4];
                // SWAR SUM: Fast horizontal byte addition
                currentSum += (chunk & 0xFF) + ((chunk >> 8) & 0xFF) + 
                              ((chunk >> 16) & 0xFF) + ((chunk >> 24) & 0xFF);
            }
            finalSum = currentSum;
        }
        floor[floorIdx] = finalSum; 
    }
}

int main() {
    const int N = 512; 
    const size_t total = (size_t)N * N * N;
    const int iter = 100;
    
    uint8_t *grid;
    int *floor;

    cudaMallocManaged(&grid, total);
    cudaMallocManaged(&floor, N * N * sizeof(int));

    for (size_t i = 0; i < total; i++) grid[i] = (i % 7 == 0) ? 1 : 0;

    cudaEvent_t start, stop;
    float timeTraditional, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    std::cout << "--- N=512 UNROLLED ASCENSION ---" << std::endl;

    // 1. TRADITIONAL (100 Kernel Launches, 1 iteration each)
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++) {
        collapseKernel<<<blocks, threads>>>((uint32_t*)grid, floor, N, 1);
    }
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeTraditional, start, stop);

    // 2. DNA PERSISTENT (1 Kernel Launch, 100 iterations)
    cudaEventRecord(start);
    collapseKernel<<<blocks, threads>>>((uint32_t*)grid, floor, N, iter);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeDNA, start, stop);

    std::cout << "Traditional: " << timeTraditional << " ms" << std::endl;
    std::cout << "DNA Unrolled: " << timeDNA << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "FINAL GAP: " << timeTraditional / timeDNA << "x" << std::endl;

    cudaFree(grid); cudaFree(floor);
    return 0;
}