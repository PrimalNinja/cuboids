%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void persistentNeuralKernel(int8_t* signal, int8_t* weights, int8_t* bias, int8_t* out, int N, int iterations) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    if (x < N && y < N && z < N) {
        int idx = x + (y * N) + (z * N * N);
        int8_t s = signal[idx];
        int8_t w = weights[idx];
        int8_t b = bias[idx];

        for (int i = 0; i < iterations; i++) {
            // THE NEURAL PULSE: (Signal * Weight) + Bias (mod 3)
            s = ( (s * w) + b ) % 3;
            if (s < 0) s += 3; // Ensure ternary wrap
        }
        out[idx] = s;
    }
}

int main() {
    const int N = 10;
    const int total = N * N * N;
    const int iterations = 1000000;
    int8_t *d_s, *d_w, *d_b, *d_out;

    cudaMallocManaged(&d_s, total);
    cudaMallocManaged(&d_w, total);
    cudaMallocManaged(&d_b, total);
    cudaMallocManaged(&d_out, total);

    for (int i = 0; i < total; i++) { d_s[i] = 1; d_w[i] = -1; d_b[i] = 1; }

    std::cout << "--- 21/12 RELEASE: FILE 0042 (PERSISTENT SYNAPSE) ---" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    persistentNeuralKernel<<<1, dim3(N, N, N)>>>(d_s, d_w, d_b, d_out, N, iterations);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Neural Pulse Time: " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << (double)iterations * total / (ms / 1000.0) / 1e9 << " GSynapse-Ops/s" << std::endl;

    cudaFree(d_s); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
    return 0;
}