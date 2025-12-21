%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 32
#define BATCH 64
#define VOLUME (N * N * N)
#define TOTAL (VOLUME * BATCH)

// --- TRADITIONAL: FP32 Batch Processing (4 bytes per voxel) ---
// Total Memory pressure per call: ~16.8 MB (In + Out)
__global__ void traditionalBatch(float* in, float* weights, float* out) {
    int b = blockIdx.y;
    int voxelIdx = threadIdx.x + (threadIdx.y * N) + (blockIdx.x * N * N);
    int batchOffset = b * VOLUME;
    out[batchOffset + voxelIdx] = in[batchOffset + voxelIdx] * weights[voxelIdx];
}

// --- DNA PERSISTENT: Ternary Batch Processing (1 byte per voxel) ---
// Total Memory pressure per call: ~4.2 MB (In + Out)
__global__ void dnaBatch(int8_t* in, int8_t* weights, int8_t* out) {
    int b = blockIdx.y;
    int voxelIdx = threadIdx.x + (threadIdx.y * N) + (blockIdx.x * N * N);
    int batchOffset = b * VOLUME;
    out[batchOffset + voxelIdx] = in[batchOffset + voxelIdx] * weights[voxelIdx];
}

int main() {
    // Allocation for Traditional
    float *d_inF, *d_wF, *d_outF;
    cudaMallocManaged(&d_inF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_wF, VOLUME * sizeof(float));
    cudaMallocManaged(&d_outF, TOTAL * sizeof(float));

    // Allocation for DNA
    int8_t *d_inT, *d_wT, *d_outT;
    cudaMallocManaged(&d_inT, TOTAL);
    cudaMallocManaged(&d_wT, VOLUME);
    cudaMallocManaged(&d_outT, TOTAL);

    dim3 threads(N, N, 1);
    dim3 blocks(N, BATCH, 1);

    std::cout << "--- 21/12 RELEASE: FILE 0052 (SWARM THROUGHPUT) ---" << std::endl;

    // 1. TRADITIONAL Race
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) traditionalBatch<<<blocks, threads>>>(d_inF, d_wF, d_outF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Race
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) dnaBatch<<<blocks, threads>>>(d_inT, d_wT, d_outT);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (FP32 Swarm): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Ternary):  " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "POPULATION SPEEDUP: " << (trad_ms / dna_ms) << "x" << std::endl;
    std::cout << "Throughput: " << (long long)TOTAL * 1000 / (dna_ms / 1000.0) / 1e9 << " GigaVoxels/sec" << std::endl;

    cudaFree(d_inF); cudaFree(d_wF); cudaFree(d_outF);
    cudaFree(d_inT); cudaFree(d_wT); cudaFree(d_outT);
    return 0;
}