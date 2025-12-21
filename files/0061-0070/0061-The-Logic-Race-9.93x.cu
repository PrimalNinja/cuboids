%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000 // 1 Million Logic Operations

// --- TRADITIONAL: Floating Point Multiplication ---
__global__ void traditionalMath(float* a, float* b, float* res) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < N) {
        atomicAdd(res, a[idx] * b[idx]);
    }
}

// --- DNA PERSISTENT: Ternary Identity Logic ---
__device__ int8_t ternaryMatch(int8_t a, int8_t b) {
    if (a == 0 || b == 0) return 0;
    return (a == b) ? 1 : -1;
}

__global__ void dnaLogic(int8_t* a, int8_t* b, int* res) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < N) {
        int8_t match = ternaryMatch(a[idx], b[idx]);
        atomicAdd(res, (int)match);
    }
}

int main() {
    float *d_aF, *d_bF, *d_resF;
    int8_t *d_aT, *d_bT;
    int *d_resT;

    cudaMallocManaged(&d_aF, N * sizeof(float));
    cudaMallocManaged(&d_bF, N * sizeof(float));
    cudaMallocManaged(&d_resF, sizeof(float));
    cudaMallocManaged(&d_aT, N);
    cudaMallocManaged(&d_bT, N);
    cudaMallocManaged(&d_resT, sizeof(int));

    std::cout << "--- 21/12 RELEASE: FILE 0061 (LOGIC VS MATH) ---" << std::endl;

    // 1. TRADITIONAL Race (FMA math)
    auto s1 = std::chrono::high_resolution_clock::now();
    traditionalMath<<<(N+255)/256, 256>>>(d_aF, d_bF, d_resF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Race (Identity Logic)
    auto s2 = std::chrono::high_resolution_clock::now();
    dnaLogic<<<(N+255)/256, 256>>>(d_aT, d_bT, d_resT);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (FP32 Multiplication): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Ternary Logic):    " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "LOGICAL EFFICIENCY GAIN: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_aF); cudaFree(d_bF); cudaFree(d_resF);
    cudaFree(d_aT); cudaFree(d_bT); cudaFree(d_resT);
    return 0;
}