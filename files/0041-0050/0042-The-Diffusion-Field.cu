%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 10

__global__ void diffusionKernel(int8_t* grid, int8_t* out, int iterations) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    int idx = x + (y * N) + (z * N * N);

    // Local state residency
    int8_t localVal = grid[idx];

    for (int i = 0; i < iterations; i++) {
        // SIMULATED DIFFUSION:
        // In a real DNA loop, we'd use Shared Memory for neighbors.
        // For this stress test, we'll perform a "Self-Influence" shift.
        localVal = (localVal + (localVal << 1)) % 3;
        if (localVal < 0) localVal += 3;
    }
    out[idx] = localVal;
}

int main() {
    const int total = N * N * N;
    const int iterations = 1000000;
    int8_t *d_grid, *d_out;

    cudaMallocManaged(&d_grid, total);
    cudaMallocManaged(&d_out, total);

    for (int i = 0; i < total; i++) d_grid[i] = 1;

    std::cout << "--- 21/12 RELEASE: FILE 0043 (VOLUMETRIC DIFFUSION) ---" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    diffusionKernel<<<1, dim3(N, N, N)>>>(d_grid, d_out, iterations);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Diffusion Time: " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << (double)iterations * total / (ms / 1000.0) / 1e9 << " Giga-Diffusions/s" << std::endl;

    cudaFree(d_grid); cudaFree(d_out);
    return 0;
}