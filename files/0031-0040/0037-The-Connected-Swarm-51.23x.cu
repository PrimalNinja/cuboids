%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define BRAINS 16
#define VOXELS 1000

// --- TRADITIONAL: Isolated Iterations ---
__global__ void traditionalKernel(int8_t* A, int8_t* B, int8_t* out) {
    int b = blockIdx.y;
    int tid = threadIdx.x;
    if (tid < VOXELS) {
        int idx = b * VOXELS + tid;
        out[idx] = (A[idx] + B[idx]) % 3;
    }
}

// --- DNA PERSISTENT: Cross-Talk Swarm ---
__global__ void dnaPersistentKernel(int8_t* A, int8_t* B, int8_t* out, int iterations) {
    // Shared memory for "Cross-Talk" within the SM
    __shared__ int8_t exchange[BRAINS];
    
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int idx = b * VOXELS + tid;

    if (tid < VOXELS) {
        int8_t localA = A[idx];
        int8_t localB = B[idx];

        for (int i = 0; i < iterations; i++) {
            // Internal logic
            localA = (localA + localB) % 3;

            // CROSS-TALK: Every 1000 cycles, sync with the neighbor brain
            if (i % 1000 == 0) {
                if (tid == 0) exchange[b] = localA; // Brain leader posts state
                __syncthreads();
                
                // Pull from neighbor (Circular shift)
                int neighbor = (b + 1) % BRAINS;
                localB = (localB + exchange[neighbor]) % 3;
                __syncthreads();
            }
        }
        out[idx] = localA;
    }
}

int main() {
    const int total = BRAINS * VOXELS;
    const int iterations = 1000000;
    int8_t *gridA, *gridB, *gridOut;

    cudaMallocManaged(&gridA, total);
    cudaMallocManaged(&gridB, total);
    cudaMallocManaged(&gridOut, total);

    for (int i = 0; i < total; i++) { gridA[i] = 1; gridB[i] = 1; }

    dim3 threads(VOXELS);
    dim3 blocks(1, BRAINS);

    std::cout << "--- 21/12 RELEASE: FILE 0037 (CONNECTED SWARM) ---" << std::endl;

    // 1. TRADITIONAL
    auto s1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        traditionalKernel<<<blocks, threads>>>(gridA, gridB, gridOut);
    }
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();

    // 2. DNA PERSISTENT
    auto s2 = std::chrono::high_resolution_clock::now();
    dnaPersistentKernel<<<blocks, threads>>>(gridA, gridB, gridOut, iterations);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();

    float trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();
    float dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (Isolated): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Swarm): " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "EFFICIENCY GAP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(gridA); cudaFree(gridB); cudaFree(gridOut);
    return 0;
}