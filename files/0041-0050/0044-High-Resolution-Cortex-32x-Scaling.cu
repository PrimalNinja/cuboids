%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 32

// --- TRADITIONAL: 1 Million CPU-Driven Launches ---
__global__ void traditionalBrain(int8_t* in, int8_t* w, int8_t* out) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    int8_t signal = in[idx] * w[idx];
    out[idx] = (signal > 0) ? 1 : (signal < 0) ? -1 : 0;
}

// --- DNA PERSISTENT: 1 Million Silicon-Resident Cycles ---
__global__ void dnaBrain(int8_t* in, int8_t* w, int8_t* out, int iterations) {
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    
    // Register-Resident States
    int8_t s = in[idx];
    int8_t weight = w[idx];

    for (int i = 0; i < iterations; i++) {
        // High-Speed Ternary Pulse
        s = (s * weight);
        s = (s > 0) ? 1 : (s < 0) ? -1 : 0;
        
        // Simulate a simple feedback flip to keep the brain "active"
        if (i % 2 == 0) s = -s; 
    }
    out[idx] = s;
}

int main() {
    const int total = N * N * N;
    const int iterations = 1000000;
    int8_t *d_in, *d_w, *d_out;

    cudaMallocManaged(&d_in, total);
    cudaMallocManaged(&d_w, total);
    cudaMallocManaged(&d_out, total);

    for (int i = 0; i < total; i++) { d_in[i] = 1; d_w[i] = -1; }

    dim3 threads(N, N, 1);
    dim3 blocks(1, 1, N);

    std::cout << "--- 21/12 RELEASE: FILE 0044 (32x32x32 CORTEX) ---" << std::endl;

    // 1. TRADITIONAL BENCHMARK
    auto s1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) { // Running 1k instead of 1M to save time, then scaling
        traditionalBrain<<<blocks, threads>>>(d_in, d_w, d_out);
    }
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count() * 1000.0; // Scaled to 1M

    // 2. DNA PERSISTENT BENCHMARK
    auto s2 = std::chrono::high_resolution_clock::now();
    dnaBrain<<<blocks, threads>>>(d_in, d_w, d_out, iterations);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (1M Projected): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (1M Actual):  " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "EFFICIENCY GAP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    return 0;
}