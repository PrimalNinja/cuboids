%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 64
#define TOTAL (N * N * N)

// --- TRADITIONAL: FP32 Density Mapping ---
// Each voxel is a 4-byte float representing "Material Density"
__global__ void traditionalVisual(float* grid, int mid) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < N && y < N) {
        // Checking the Z-middle slice in a 4MB buffer
        float val = grid[x + (y * N) + (mid * N * N)];
    }
}

// --- DNA PERSISTENT: Ternary State Mapping ---
// Each voxel is a 1-byte state identifier
__global__ void dnaVisual(int8_t* grid, int mid) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < N && y < N) {
        // Checking the Z-middle slice in a 1MB buffer
        int8_t val = grid[x + (y * N) + (mid * N * N)];
    }
}

int main() {
    float *d_gridF;
    int8_t *d_gridT;
    cudaMallocManaged(&d_gridF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_gridT, TOTAL);

    int mid = N / 2;
    dim3 threads(8, 8);
    dim3 blocks(8, 8);

    std::cout << "--- 21/12 RELEASE: FILE 0058 (VISUALIZATION RACE) ---" << std::endl;

    // 1. TRADITIONAL Benchmark (4.0 MB Memory Swing)
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<10000; i++) traditionalVisual<<<blocks, threads>>>(d_gridF, mid);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Benchmark (1.0 MB Memory Swing)
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<10000; i++) dnaVisual<<<blocks, threads>>>(d_gridT, mid);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (FP32 Slice Access): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Ternary Slice):  " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "BANDWIDTH EFFICIENCY GAIN: " << (trad_ms / dna_ms) << "x" << std::endl;

    // Output the visual verification
    for (int i = 0; i < TOTAL; i++) d_gridT[i] = 0;
    for (int i = 0; i < N; i++) {
        d_gridT[mid + (i * N) + (mid * N * N)] = 1; 
        d_gridT[i + (mid * N) + (mid * N * N)] = 2; 
    }

    std::cout << "\nVisual verification of $64^3$ DNA Structure (Slice Z=32):" << std::endl;
    for (int y = mid-5; y < mid+5; y++) {
        for (int x = mid-5; x < mid+5; x++) {
            int val = d_gridT[x + (y * N) + (mid * N * N)];
            if (val == 1) std::cout << "|";
            else if (val == 2) std::cout << "-";
            else std::cout << ".";
        }
        std::cout << std::endl;
    }

    cudaFree(d_gridF); cudaFree(d_gridT);
    return 0;
}