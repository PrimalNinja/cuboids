%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 10

// --- TRADITIONAL: Legacy VRAM-Bound Diffusion ---
__global__ void traditionalDiffusion(int8_t* grid, int8_t* out) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    int idx = x + (y * N) + (z * N * N);

    // Legacy must read from global memory every single time
    int8_t val = grid[idx];
    out[idx] = (val + (val << 1)) % 3;
}

// --- DNA PERSISTENT: Silicon-Resident Diffusion ---
__global__ void dnaDiffusion(int8_t* grid, int8_t* out, int iterations) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    int idx = x + (y * N) + (z * N * N);

    // Resident state: Stays in Registers for the entire million cycles
    int8_t localVal = grid[idx];

    for (int i = 0; i < iterations; i++) {
        localVal = (localVal + (localVal << 1)) % 3;
        if (localVal < 0) localVal += 3;
    }
    out[idx] = localVal;
}

int main() {
    const int total = N * N * N;
    const int iterations = 1000000;
    int8_t *d_grid, *d_out;

    cudaMallocManaged(&d_grid, total);
    cudaMallocManaged(&d_out, total);

    for (int i = 0; i < total; i++) d_grid[i] = 1;

    std::cout << "--- 21/12 RELEASE: FILE 0043 (DIFFUSION RACE) ---" << std::endl;

    // 1. TRADITIONAL Benchmark
    auto s1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        traditionalDiffusion<<<1, dim3(N, N, N)>>>(d_grid, d_out);
    }
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    float trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Benchmark
    auto s2 = std::chrono::high_resolution_clock::now();
    dnaDiffusion<<<1, dim3(N, N, N)>>>(d_grid, d_out, iterations);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    float dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (1M VRAM R/W): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Register): " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "EFFICIENCY GAP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_grid); cudaFree(d_out);
    return 0;
}