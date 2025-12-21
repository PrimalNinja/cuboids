%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define N 64
#define TOTAL_VOXELS (N * N * N)

// --- Sparse DNA Signal Structure ---
// Instead of a massive 3D grid, we only store the index and the state.
struct SparseSynapse {
    unsigned short address; // 16-bit location
    int8_t state;           // Ternary logic (-1, 0, 1)
};

// --- KERNEL: SYNAPTIC DISTILLATION ---
// This kernel scans the high-res manifold and extracts ONLY the active neurons.
__global__ void distillSynapsesKernel(int8_t* brain, int* activeCount, SparseSynapse* output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    // If the brain is empty here, we ignore it. This is where the efficiency comes from.
    if (brain[tid] != 0) {
        int idx = atomicAdd(activeCount, 1);
        output[idx].address = (unsigned short)tid;
        output[idx].state = brain[tid];
    }
}

int main() {
    int8_t *d_brain;
    int *d_count, h_count = 0;
    SparseSynapse *d_export;

    cudaMallocManaged(&d_brain, TOTAL_VOXELS);
    cudaMallocManaged(&d_count, sizeof(int));
    cudaMallocManaged(&d_export, TOTAL_VOXELS * sizeof(SparseSynapse));

    // MOCK: A winner with 100 active connections
    cudaMemset(d_brain, 0, TOTAL_VOXELS);
    for(int i=0; i<100; i++) d_brain[(i * 1337) % TOTAL_VOXELS] = 1;

    cudaMemset(d_count, 0, sizeof(int));

    // --- RUN DISTILLATION ---
    distillSynapsesKernel<<<TOTAL_VOXELS/256+1, 256>>>(d_brain, d_count, d_export);
    cudaDeviceSynchronize();
    h_count = *d_count;

    // --- TRADITIONAL VS SPARSE TELEMETRY ---
    size_t tradSize = TOTAL_VOXELS; // Traditional: 1 byte per voxel, even the zeros
    size_t sparseSize = h_count * sizeof(SparseSynapse); // DNA: Only the "sparks"

    std::cout << "--- 0068 STORAGE PERFORMANCE PROFILE ---" << std::endl;
    std::cout << "Traditional Dense Payload: " << tradSize / 1024 << " KB" << std::endl;
    std::cout << "Sparse DNA Payload:        " << (float)sparseSize / 1024.0f << " KB" << std::endl;
    std::cout << "Bandwidth Efficiency:      " << (float)tradSize / sparseSize << "x Lighter" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "First 5 Active Synapses: ";
    for(int i=0; i<5; i++) printf("[%04X:%d] ", d_export[i].address, d_export[i].state);
    std::cout << "..." << std::endl;

    cudaFree(d_brain); cudaFree(d_count); cudaFree(d_export);
    return 0;
}