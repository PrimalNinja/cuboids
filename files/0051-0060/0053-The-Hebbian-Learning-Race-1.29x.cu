%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 32
#define TOTAL (N * N * N)

// --- TRADITIONAL: FP32 Gradient Descent Update ---
// Mimics a standard AI weight update: W = W + (input * learning_rate)
__global__ void traditionalLearn(float* input, float* weights, float lr) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    weights[idx] += input[idx] * lr; // Floating point multiply + add
}

// --- DNA PERSISTENT: Ternary Hebbian Update ---
// Direct signal absorption: If Weight is 0, take the Input's state.
__global__ void dnaLearn(int8_t* input, int8_t* weights) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    if (input[idx] != 0 && weights[idx] == 0) {
        weights[idx] = input[idx]; 
    }
}

int main() {
    float *d_inF, *d_wF;
    int8_t *d_inT, *d_wT;

    cudaMallocManaged(&d_inF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_wF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_inT, TOTAL);
    cudaMallocManaged(&d_wT, TOTAL);

    dim3 threads(N, N, 1);
    dim3 blocks(1, 1, N);

    std::cout << "--- 21/12 RELEASE: FILE 0053 (HEBBIAN RACE) ---" << std::endl;

    // 1. TRADITIONAL Race (Arithmetic Gradient)
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) traditionalLearn<<<blocks, threads>>>(d_inF, d_wF, 0.01f);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Race (Logical Absorption)
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) dnaLearn<<<blocks, threads>>>(d_inT, d_wT);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (FP32 Gradient): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Logical):    " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "LEARNING EFFICIENCY GAP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_inF); cudaFree(d_wF);
    cudaFree(d_inT); cudaFree(d_wT);
    return 0;
}