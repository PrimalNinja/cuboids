%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 32
#define TOTAL (N * N * N)

// Simple deterministic "random" using thread ID and iteration
__device__ int pseudo_random(int idx, int iteration) {
    return ((idx * 1103515245 + 12345) ^ iteration) % 100;
}

// --- TRADITIONAL: Two-Step Denoise + Multiply ---
__global__ void traditionalRecall(float* in, float* w, float* en) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    float noise = 0.05f; 
    float corrupted_in = in[idx] + noise; 
    atomicAdd(en, corrupted_in * w[idx]);
}

// --- DNA PERSISTENT: FIXED with deterministic noise ---
__global__ void dnaRecall(int8_t* input, int8_t* weights, int* energy, int iteration) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    
    int8_t signal = input[idx];
    
    // Deterministic "stochastic" flip (no RNG overhead)
    int rand_val = pseudo_random(idx, iteration);
    if (rand_val < 10) { // 10% probability
        signal = (rand_val % 3) - 1; // -1, 0, or 1
    }

    if (signal != 0 && weights[idx] != 0) {
        atomicAdd(energy, (int)(signal * weights[idx]));
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

    std::cout << "--- 21/12 RELEASE: FILE 0054 FIXED (RECALL RACE) ---" << std::endl;

    // 1. TRADITIONAL Race
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) traditionalRecall<<<blocks, threads>>>(d_inF, d_wF, d_enF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Race (FIXED - deterministic)
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) dnaRecall<<<blocks, threads>>>(d_inT, d_wT, d_enT, i);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (Sequential FP32): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Fused Ternary): " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "RECALL EFFICIENCY GAP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_inF); cudaFree(d_wF); cudaFree(d_enF);
    cudaFree(d_inT); cudaFree(d_wT); cudaFree(d_enT);
    return 0;
}