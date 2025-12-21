%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void matchKernel(int8_t* input, int8_t* target, int* energy, int N) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = blockIdx.z;
    int idx = x + (y * N) + (z * N * N);
    
    // TERNARY CORRELATION LOGIC:
    // 1. If states match and are non-zero: Resonance (+1 Energy)
    // 2. If states conflict (1 vs -1): Dissonance (-1 Energy)
    // 3. If either is 0: Neutrality (0 Energy)
    
    if (input[idx] == target[idx] && input[idx] != 0) {
        atomicAdd(energy, 1); 
    } else if (input[idx] != target[idx] && input[idx] != 0 && target[idx] != 0) {
        atomicAdd(energy, -1);
    }
}

int main() {
    const int N = 32;
    const int total = N * N * N;
    int8_t *d_input, *d_target;
    int *d_energy;

    cudaMallocManaged(&d_input, total);
    cudaMallocManaged(&d_target, total);
    cudaMallocManaged(&d_energy, sizeof(int));

    *d_energy = 0;

    // Zero out the field
    for (int i = 0; i < total; i++) { d_input[i] = 0; d_target[i] = 0; }
    
    // Inject a specific 10-voxel "Signal" into both Input and Target
    for (int i = 0; i < 10; i++) {
        d_input[i] = 1; 
        d_target[i] = 1; 
    }

    // Launch a 3D grid of 32,768 threads
    dim3 threads(N, N, 1);
    dim3 blocks(1, 1, N);

    std::cout << "--- 21/12 RELEASE: FILE 0046 (RESONANCE MATCHER) ---" << std::endl;
    matchKernel<<<blocks, threads>>>(d_input, d_target, d_energy, N);
    cudaDeviceSynchronize();

    std::cout << "Detected Pattern Energy: " << *d_energy << std::endl;
    if (*d_energy == 10) std::cout << "STATUS: SIGNAL MATCHED" << std::endl;

    cudaFree(d_input); cudaFree(d_target); cudaFree(d_energy);
    return 0;
}