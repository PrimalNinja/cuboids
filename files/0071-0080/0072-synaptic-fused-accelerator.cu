%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 64
#define BATCH_SIZE 64
#define TOTAL_VOXELS (1ULL * N * N * N)
#define GEN_COUNT 50

// DNA: 1-byte per voxel, Ternary logic
__global__ void dnaPulse(int8_t* target, int8_t* brains, int* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < TOTAL_VOXELS) {
        int8_t res = (target[tid] == brains[b * TOTAL_VOXELS + tid]) ? 1 : -1;
        atomicAdd(&energies[b], (int)res);
    }
}

// TRADITIONAL: 4-bytes per voxel, FP32 logic
__global__ void tradPulse(float* target, float* brains, float* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < TOTAL_VOXELS) {
        float res = target[tid] * brains[b * TOTAL_VOXELS + tid];
        atomicAdd(&energies[b], res);
    }
}

int main() {
    int8_t *d_tDNA, *d_bDNA; int *d_eDNA;
    float *d_tTrad, *d_bTrad, *d_eTrad;

    cudaMalloc(&d_tDNA, TOTAL_VOXELS);
    cudaMalloc(&d_bDNA, BATCH_SIZE * TOTAL_VOXELS);
    cudaMalloc(&d_eDNA, BATCH_SIZE * sizeof(int));

    cudaMalloc(&d_tTrad, TOTAL_VOXELS * sizeof(float));
    cudaMalloc(&d_bTrad, BATCH_SIZE * TOTAL_VOXELS * sizeof(float));
    cudaMalloc(&d_eTrad, BATCH_SIZE * sizeof(float));

    cudaEvent_t s1, e1, s2, e2;
    cudaEventCreate(&s1); cudaEventCreate(&e1);
    cudaEventCreate(&s2); cudaEventCreate(&e2);

    // RACE 1: DNA (int8)
    cudaEventRecord(s1);
    for(int g=0; g<GEN_COUNT; g++) {
        dnaPulse<<<dim3(TOTAL_VOXELS/256+1, BATCH_SIZE), 256>>>(d_tDNA, d_bDNA, d_eDNA);
    }
    cudaEventRecord(e1);

    // RACE 2: TRADITIONAL (float32)
    cudaEventRecord(s2);
    for(int g=0; g<GEN_COUNT; g++) {
        tradPulse<<<dim3(TOTAL_VOXELS/256+1, BATCH_SIZE), 256>>>(d_tTrad, d_bTrad, d_eTrad);
    }
    cudaEventRecord(e2);

    cudaEventSynchronize(e1); cudaEventSynchronize(e2);
    float dna_ms, trad_ms;
    cudaEventElapsedTime(&dna_ms, s1, e1);
    cudaEventElapsedTime(&trad_ms, s2, e2);

    // 64-bit calculation for display
    unsigned long long totalOps = TOTAL_VOXELS * BATCH_SIZE * GEN_COUNT;

    std::cout << "DNA PULSE (int8): " << dna_ms << " ms | " << (double)totalOps / (dna_ms/1000.0) / 1e9 << " GVox/s" << std::endl;
    std::cout << "TRAD PULSE (f32):  " << trad_ms << " ms | " << (double)totalOps / (trad_ms/1000.0) / 1e9 << " GVox/s" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "RAW COMPUTE GAP: " << (trad_ms / dna_ms) << "x" << std::endl;

    return 0;
}