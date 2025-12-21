%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define BATCH 64

// --- TRADITIONAL: Softmax Selection ---
// Legacy AI often calculates probabilities for every candidate
__global__ void traditionalSelection(float* rewards, float* probabilities) {
    if (threadIdx.x == 0) {
        float sum = 0;
        for(int i=0; i<BATCH; i++) sum += expf(rewards[i]);
        for(int i=0; i<BATCH; i++) probabilities[i] = expf(rewards[i]) / sum;
    }
}

// --- DNA PERSISTENT: Darwinian Selection ---
// Pure integer Max-Reduction (All-or-Nothing survival)
__global__ void dnaSelection(int* energies, int* winnerIdx) {
    if (threadIdx.x == 0) {
        int bestVal = -999999;
        int bestIdx = 0;
        for (int i = 0; i < BATCH; i++) {
            if (energies[i] > bestVal) {
                bestVal = energies[i];
                bestIdx = i;
            }
        }
        *winnerIdx = bestIdx;
    }
}

int main() {
    float *d_rewF, *d_probF;
    int *d_enT, *d_winT;
    
    cudaMallocManaged(&d_rewF, BATCH * sizeof(float));
    cudaMallocManaged(&d_probF, BATCH * sizeof(float));
    cudaMallocManaged(&d_enT, BATCH * sizeof(int));
    cudaMallocManaged(&d_winT, sizeof(int));

    std::cout << "--- 21/12 RELEASE: FILE 0060 (SELECTION RACE) ---" << std::endl;

    // 1. TRADITIONAL Race (Probabilistic / Exponential)
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100000; i++) traditionalSelection<<<1, 1>>>(d_rewF, d_probF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Race (Deterministic / Integer)
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100000; i++) dnaSelection<<<1, 1>>>(d_enT, d_winT);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (Softmax Selection): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Darwinian):      " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "DECISION SPEEDUP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_rewF); cudaFree(d_probF);
    cudaFree(d_enT); cudaFree(d_winT);
    return 0;
}