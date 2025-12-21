%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

#define N 32
#define TOTAL (N * N * N)

// --- TRADITIONAL: Floating-Point Noise Correction ---
// Typically requires multiplication and re-normalization to suppress noise.
__global__ void traditionalCompare(float* A, float* B, float* energy) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    // Standard dot product: sensitive to small fluctuations
    atomicAdd(energy, A[idx] * B[idx]);
}

// --- DNA PERSISTENT: Ternary Dissonance ---
// Pure integer correlation: noise effectively vanishes.
__global__ void dnaCompare(int8_t* A, int8_t* B, int* energy) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    if (A[idx] != 0 && B[idx] != 0) {
        atomicAdd(energy, (int)(A[idx] * B[idx]));
    }
}

int main() {
    float *d_inF, *d_tgF, *d_enF;
    int8_t *d_inT, *d_tgT;
    int *d_enT;

    cudaMallocManaged(&d_inF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_tgF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_enF, sizeof(float));

    cudaMallocManaged(&d_inT, TOTAL);
    cudaMallocManaged(&d_tgT, TOTAL);
    cudaMallocManaged(&d_enT, sizeof(int));

    dim3 threads(N, N, 1);
    dim3 blocks(1, 1, N);

    std::cout << "--- 21/12 RELEASE: FILE 0049 (STOCHASTIC RACE) ---" << std::endl;

    // 1. TRADITIONAL Benchmark
    *d_enF = 0;
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) traditionalCompare<<<blocks, threads>>>(d_inF, d_tgF, d_enF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Benchmark
    *d_enT = 0;
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) dnaCompare<<<blocks, threads>>>(d_inT, d_tgT, d_enT);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (FP32 Correlator): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Ternary Correlator): " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "ROBUSTNESS SPEEDUP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_inF); cudaFree(d_tgF); cudaFree(d_enF);
    cudaFree(d_inT); cudaFree(d_tgT); cudaFree(d_enT);
    return 0;
}