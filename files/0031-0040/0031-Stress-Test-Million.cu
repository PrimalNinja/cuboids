%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void speedKernel(int8_t* A, int8_t* B, int8_t* out, int N) {
    int idx = threadIdx.x + (threadIdx.y * N) + (threadIdx.z * N * N);
    if (idx < (N * N * N)) {
        out[idx] = (A[idx] + B[idx]) % 3;
    }
}

int main() {
    const int N = 6;
    const int total = N * N * N;
    const int iterations = 1000000;
    int8_t *gridA, *gridB, *gridOut;

    cudaMallocManaged(&gridA, total);
    cudaMallocManaged(&gridB, total);
    cudaMallocManaged(&gridOut, total);

    for (int i = 0; i < total; i++) { gridA[i] = 1; gridB[i] = 1; }

    dim3 blockSize(N, N, N);

    std::cout << "Starting 1,000,000 iterations..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        speedKernel<<<1, blockSize>>>(gridA, gridB, gridOut, N);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "TOTAL TIME: " << elapsed.count() << " ms" << std::endl;
    std::cout << "TIME PER INTERFERENCE: " << (elapsed.count() * 1000.0) / iterations << " microseconds" << std::endl;

    cudaFree(gridA); cudaFree(gridB); cudaFree(gridOut);
    return 0;
}