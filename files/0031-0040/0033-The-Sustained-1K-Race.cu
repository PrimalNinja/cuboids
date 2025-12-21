%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// --- TRADITIONAL: The "Neighbor Tax" is paid in VRAM ---
__global__ void traditionalNeural(int8_t* A, int8_t* B, int8_t* out, int N) {
    int x = threadIdx.x; int y = threadIdx.y; int z = blockIdx.z;
    int idx = x + (y * N) + (z * N * N);

    if (x < N && y < N && z < N) {
        int8_t stateA = A[idx];
        int8_t stateB = B[idx];
        
        // Traditional neighbor check: Must read from global memory (A)
		int8_t neighbor = (x > 0) ? A[(x-1) + y * N + z * N * N] : stateB;
		out[idx] = (stateA + stateB + neighbor) % 3; // neighbor now affects output
        
        // State evolution
        int8_t res = (stateA + stateB) % 3;
        out[idx] = res;
        // In the next launch, B will be influenced by this 'neighbor' result
    }
}

// --- DNA PERSISTENT: The "Neighbor Tax" is paid in Silicon ---
__global__ void dnaNeuralKernel(int8_t* A, int8_t* B, int8_t* out, int N, int iterations) {
    extern __shared__ int8_t syncSpace[];
    int x = threadIdx.x; int y = threadIdx.y; int z = blockIdx.z;
    int idx = x + (y * N) + (z * N * N);

    int8_t sA = A[idx];
    int8_t sB = B[idx];

    for (int i = 0; i < iterations; i++) {
        sA = (sA + sB) % 3;

        // Every 1000 generations, voxels communicate
        if (i % 1000 == 0) {
            syncSpace[x + y * N] = sA;
            __syncthreads(); 
            int8_t neighbor = (x > 0) ? syncSpace[(x-1) + y * N] : sB;
            sB = (sB + neighbor) % 3; 
            __syncthreads();
        }
    }
    out[idx] = sA;
}

int main() {
    const int N = 10;
    const int total = N * N * N;
    const int iterations = 1000000;
    int8_t *gridA, *gridB, *gridOut;

    cudaMallocManaged(&gridA, total);
    cudaMallocManaged(&gridB, total);
    cudaMallocManaged(&gridOut, total);

    for (int i = 0; i < total; i++) { gridA[i] = 1; gridB[i] = 1; }

    dim3 threads(N, N, 1);
    dim3 blocks(1, 1, N);
    size_t sharedMemSize = N * N * sizeof(int8_t);

    std::cout << "--- FILE 0034: THE NEURAL GAP ---" << std::endl;

    // 1. TRADITIONAL (The Slow Road)
    auto s1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        traditionalNeural<<<blocks, threads>>>(gridA, gridB, gridOut, N);
        // In real traditional logic, we'd swap gridA and gridOut here.
    }
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> trad_ms = e1 - s1;

    // 2. DNA PERSISTENT (The Fast Road)
    auto s2 = std::chrono::high_resolution_clock::now();
    dnaNeuralKernel<<<blocks, threads, sharedMemSize>>>(gridA, gridB, gridOut, N, iterations);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dna_ms = e2 - s2;

    std::cout << "Traditional (VRAM Bound): " << trad_ms.count() << " ms" << std::endl;
    std::cout << "DNA (Silicon Bound):      " << dna_ms.count() << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "FINAL NEURAL GAP: " << (trad_ms.count() / dna_ms.count()) << "x" << std::endl;

    cudaFree(gridA); cudaFree(gridB); cudaFree(gridOut);
    return 0;
}