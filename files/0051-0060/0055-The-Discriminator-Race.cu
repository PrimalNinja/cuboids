%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 32
#define TOTAL (N * N * N)

// --- TRADITIONAL: FP32 Linear Discrimination ---
// Requires floating-point multiplication and a continuous summation
__global__ void traditionalDiscriminate(float* in, float* w, float* en) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    // Continuous dot product
    atomicAdd(en, in[idx] * w[idx]);
}

// --- DNA PERSISTENT: Ternary Integer Discrimination ---
// Uses hardware-level integer ALUs for ultra-fast polarity matching
__global__ void dnaDiscriminate(int8_t* in, int8_t* w, int* en) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    // Logical branch: only process active voxels (Sparsity)
    if (in[idx] != 0) {
        atomicAdd(en, (int)(in[idx] * w[idx]));
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

    dim3 threads(N, N, 1);
    dim3 blocks(1, 1, N);

    std::cout << "--- 21/12 RELEASE: FILE 0055 (DISCRIMINATION RACE) ---" << std::endl;

    // 1. TRADITIONAL Race (FP32 precision)
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) traditionalDiscriminate<<<blocks, threads>>>(d_inF, d_wF, d_enF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Race (Ternary Logic)
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) dnaDiscriminate<<<blocks, threads>>>(d_inT, d_wT, d_enT);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (FP32 SVM):     " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Ternary):   " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "DISCRIMINATION SPEEDUP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_inF); cudaFree(d_wF); cudaFree(d_enF);
    cudaFree(d_inT); cudaFree(d_wT); cudaFree(d_enT);
    return 0;
}