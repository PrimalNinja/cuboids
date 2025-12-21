%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>

// Using template <int MODE> forces the compiler to create 3 separate, 
// perfectly optimized versions of the kernel with zero branching.
template <int MODE>
__global__ void spatialTransformKernel(uint8_t* in, uint8_t* out, int N, int iterations) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    if (x < N && y < N && z < N) {
        int oldIdx = x + (y * N) + (z * N * N);
        uint8_t voxel = in[oldIdx];
        int nx = x, ny = y, nz = z;

        // DNA PERSISTENCE: Zero-Branch Geometry
        for(int i = 0; i < iterations; i++) {
            int temp;
            if (MODE == 0) { temp = ny; ny = nz; nz = (N - 1) - temp; }
            else if (MODE == 1) { temp = nx; nx = (N - 1) - nz; nz = temp; }
            else if (MODE == 2) { temp = nx; nx = ny; ny = (N - 1) - temp; }
        }

        int newIdx = nx + (ny * N) + (nz * N * N);
        out[newIdx] = voxel;
    }
}

int main() {
    const int N = 6;
    const int total = N * N * N;
    const int iter = 100;
    uint8_t *gridA, *gridB;

    cudaMallocManaged(&gridA, total);
    cudaMallocManaged(&gridB, total);

    for (int i = 0; i < total; i++) gridA[i] = 0;
    gridA[1 + (2 * N) + (3 * N * N)] = 2; // (1,2,3)

    cudaEvent_t start, stop;
    float timeTraditional, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    std::cout << "--- 21/12 TEMPLATE OPTIMIZED: MODE 1 (Y-AXIS) ---" << std::endl;

    // 1. TRADITIONAL
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++) {
        spatialTransformKernel<1><<<1, dim3(N,N,N)>>>(gridA, gridB, N, 1);
    }
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeTraditional, start, stop);

    // 2. DNA PERSISTENT
    cudaEventRecord(start);
    spatialTransformKernel<1><<<1, dim3(N,N,N)>>>(gridA, gridB, N, iter);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeDNA, start, stop);

    std::cout << "Traditional Performance: " << timeTraditional << " ms" << std::endl;
    std::cout << "DNA Persistent Performance: " << timeDNA << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "PERFORMANCE GAP: " << timeTraditional / timeDNA << "x" << std::endl;

    cudaFree(gridA); cudaFree(gridB);
    return 0;
}