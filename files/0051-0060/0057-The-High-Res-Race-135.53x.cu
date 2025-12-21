%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 64
#define TOTAL (N * N * N)

// --- TRADITIONAL: FP32 High-Resolution Filter ---
// 1 MB per volume. Higher risk of cache eviction.
__global__ void traditionalHighRes(float* in, float* w, float* en) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < N && y < N && z < N) {
        int idx = x + (y * N) + (z * N * N);
        atomicAdd(en, in[idx] * w[idx]);
    }
}

// --- DNA PERSISTENT: Ternary High-Resolution Filter ---
// 256 KB per volume. Fits entirely in L1/L2 cache.
__global__ void dnaHighRes(int8_t* in, int8_t* w, int* en) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < N && y < N && z < N) {
        int idx = x + (y * N) + (z * N * N);
        if (in[idx] != 0 && w[idx] != 0) {
            atomicAdd(en, (int)(in[idx] * w[idx]));
        }
    }
}

int main() {
    float *d_inF, *d_wF, *d_enF;
    int8_t *d_inT, *d_wT;
    int *d_enT;

    cudaMallocManaged(&d_inF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_wF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_enF, sizeof(float));
    cudaMallocManaged(&d_inT, TOTAL);
    cudaMallocManaged(&d_wT, TOTAL);
    cudaMallocManaged(&d_enT, sizeof(int));

    // Coverage: 64x64x64 mapped by 8x8x8 blocks
    dim3 threads(8, 8, 8);
    dim3 blocks(8, 8, 8);

    std::cout << "--- 21/12 RELEASE: FILE 0057 (HIGH-RES RACE) ---" << std::endl;

    // 1. TRADITIONAL Benchmark
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) traditionalHighRes<<<blocks, threads>>>(d_inF, d_wF, d_enF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Benchmark
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) dnaHighRes<<<blocks, threads>>>(d_inT, d_wT, d_enT);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (FP32 64^3): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Ternary 64^3): " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "RESOLUTION EFFICIENCY GAIN: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_inF); cudaFree(d_wF); cudaFree(d_enF);
    cudaFree(d_inT); cudaFree(d_wT); cudaFree(d_enT);
    return 0;
}