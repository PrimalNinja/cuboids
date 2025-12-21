%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void mutateKernel(int8_t* championWeights, int8_t* batchWeights, int N, unsigned long seed) {
    int b = blockIdx.y; // Batch index
    int idx = threadIdx.x + (threadIdx.y * N) + (blockIdx.z * N * N);
    int totalIdx = b * (N * N * N) + idx;

    curandState state;
    curand_init(seed + b, idx, 0, &state);

    // Copy the champion's "genes"
    int8_t gene = championWeights[idx];

    // 1% chance to mutate a weight to a random ternary value (-1, 0, 1)
    if (curand_uniform(&state) < 0.01f) {
        gene = (int8_t)((curand_uniform(&state) * 3) - 1);
    }

    batchWeights[totalIdx] = gene;
}

int main() {
    const int N = 64;
    const int volume = N * N * N;
    const int batchSize = 64;
    
    int8_t *d_champion, *d_batch;
    cudaMallocManaged(&d_champion, volume);
    cudaMallocManaged(&d_batch, volume * batchSize);

    // Initialize champion with a simple seed (e.g., all 0s)
    for (int i = 0; i < volume; i++) d_champion[i] = 0;

    dim3 threads(8, 8, 8);
    dim3 blocks(N/8, batchSize, N/8); 

    std::cout << "Evolving 64 generations of the high-res brain..." << std::endl;
    
    mutateKernel<<<blocks, threads>>>(d_champion, d_batch, N, 1234ULL);
    cudaDeviceSynchronize();

    std::cout << "Mutation Complete. 64 variant brains ready for testing." << std::endl;

    cudaFree(d_champion); cudaFree(d_batch);
    return 0;
}