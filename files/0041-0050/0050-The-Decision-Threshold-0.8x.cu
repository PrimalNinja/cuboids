%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <math.h>

// --- TRADITIONAL: Softmax-style Probabilistic Activation ---
// Requires floating point math, exponentials, and normalization
__global__ void traditionalActivation(float* energy, float* result) {
    if (threadIdx.x == 0) {
        float e = expf(*energy);
        *result = e / (e + expf(5.0f)); // Simplified softmax against a noise floor
    }
}

// --- DNA PERSISTENT: Discrete Threshold Activation ---
// Simple integer comparison (Action Potential)
__global__ void dnaActivation(int* energy, int threshold, bool* detection) {
    if (threadIdx.x == 0) {
        *detection = (*energy >= threshold);
    }
}

int main() {
    float *d_energyF, *d_resultF;
    int *d_energyI;
    bool *d_detection;
    
    cudaMallocManaged(&d_energyF, sizeof(float));
    cudaMallocManaged(&d_resultF, sizeof(float));
    cudaMallocManaged(&d_energyI, sizeof(int));
    cudaMallocManaged(&d_detection, sizeof(bool));

    *d_energyF = 10.0f;
    *d_energyI = 10;
    int threshold = 8;

    std::cout << "--- 21/12 RELEASE: FILE 0050 (ACTIVATION RACE) ---" << std::endl;

    // 1. TRADITIONAL Benchmark (Exponential Math)
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000000; i++) traditionalActivation<<<1, 1>>>(d_energyF, d_resultF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Benchmark (Logical Comparison)
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000000; i++) dnaActivation<<<1, 1>>>(d_energyI, threshold, d_detection);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (Softmax Exp): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Threshold): " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "COGNITIVE SPEEDUP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_energyF); cudaFree(d_resultF);
    cudaFree(d_energyI); cudaFree(d_detection);
    return 0;
}