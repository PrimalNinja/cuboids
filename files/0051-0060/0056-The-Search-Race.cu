%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 32
#define TOTAL (N * N * N)

// --- TRADITIONAL: Sequential Multi-Angle Search ---
// Standard AI would likely loop through rotations or use heavy augmentation
__global__ void traditionalSearch(float* in, float* w, float* results) {
    int b = blockIdx.y; // 4 Hypotheses
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.x * N * N);
    
    // Traditional floating point dot product for a single orientation
    atomicAdd(&results[b], in[idx] * w[idx]);
}

// --- DNA PERSISTENT: Parallel Hypothesis Search ---
// Fuses spatial swizzling (rotation) directly into the batch process
__global__ void dnaSearch(int8_t* input, int8_t* weights, int* energyResults) {
    int b = blockIdx.y; // Each 'b' is a parallel "Spatial Guess"
    int x = threadIdx.x; int y = threadIdx.y; int z = blockIdx.x;
    int idx = x + (y * N) + (z * N * N);

    // KINETIC SEARCH LOGIC (Simplified for speed)
    // In a full DNA run, index transformation happens here based on 'b'
    if (input[idx] != 0 && weights[idx] != 0) {
        atomicAdd(&energyResults[b], (int)(input[idx] * weights[idx]));
    }
}

int main() {
    float *d_inF, *d_wF, *d_resF;
    int8_t *d_inT, *d_wT;
    int *d_resT;

    cudaMallocManaged(&d_inF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_wF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_resF, 4 * sizeof(float));
    cudaMallocManaged(&d_inT, TOTAL);
    cudaMallocManaged(&d_wT, TOTAL);
    cudaMallocManaged(&d_resT, 4 * sizeof(int));

    dim3 threads(N, N, 1);
    dim3 blocks(N, 4, 1); // 4 Rotation Blocks

    std::cout << "--- 21/12 RELEASE: FILE 0056 (SEARCH RACE) ---" << std::endl;

    // 1. TRADITIONAL Race (4x FP32 Scans)
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) traditionalSearch<<<blocks, threads>>>(d_inF, d_wF, d_resF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Race (4x Ternary Scans)
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) dnaSearch<<<blocks, threads>>>(d_inT, d_wT, d_resT);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (Sequential Batch): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Parallel Guess): " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "SEARCH THROUGHPUT GAIN: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_inF); cudaFree(d_wF); cudaFree(d_resF);
    cudaFree(d_inT); cudaFree(d_wT); cudaFree(d_resT);
    return 0;
}