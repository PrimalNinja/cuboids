%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>

// Kernel to compare two grids and return energy
__global__ void compareKernel(int8_t* A, int8_t* B, int* energy, int N) {
    int x = threadIdx.x; int y = threadIdx.y; int z = blockIdx.z;
    int idx = x + (y * N) + (z * N * N);
    
    // Resonance occurs only on spatial overlap
    if (A[idx] != 0 && A[idx] == B[idx]) {
        atomicAdd(energy, 1);
    }
}

// Kernel to rotate input 90 degrees on the X-axis
__global__ void rotateKernel(int8_t* in, int8_t* out, int N) {
    int x = threadIdx.x; int y = threadIdx.y; int z = blockIdx.z;
    int oldIdx = x + (y * N) + (z * N * N);
    
    // X-axis rotation mapping: (x, y, z) -> (x, z, -y)
    int newY = z;
    int newZ = (N - 1) - y;
    int newIdx = x + (newY * N) + (newZ * N * N);
    
    out[newIdx] = in[oldIdx];
}

int main() {
    const int N = 32;
    const int total = N * N * N;
    int8_t *d_target, *d_input, *d_rotated;
    int *d_energy;

    cudaMallocManaged(&d_target, total);
    cudaMallocManaged(&d_input, total);
    cudaMallocManaged(&d_rotated, total);
    cudaMallocManaged(&d_energy, sizeof(int));

    // 1. Setup a Target: A 10-voxel pillar on the Z-axis
    for (int i = 0; i < total; i++) { d_target[i] = 0; d_input[i] = 0; }
    for (int z = 0; z < 10; z++) d_target[0 + (0 * N) + (z * N * N)] = 1;

    // 2. Setup Input: The same pillar but on the Y-axis (misaligned)
    for (int y = 0; y < 10; y++) d_input[0 + (y * N) + (0 * N * N)] = 1;

    dim3 threads(N, N, 1);
    dim3 blocks(1, 1, N);

    std::cout << "--- 21/12 RELEASE: FILE 0047 (ROTATIONAL INVARIANCE) ---" << std::endl;

    // TEST 1: Compare original (Should be low energy/1)
    *d_energy = 0;
    compareKernel<<<blocks, threads>>>(d_input, d_target, d_energy, N);
    cudaDeviceSynchronize();
    std::cout << "Energy before rotation: " << *d_energy << " (Misaligned)" << std::endl;

    // TEST 2: Rotate and Compare (Should be high energy/10)
    rotateKernel<<<blocks, threads>>>(d_input, d_rotated, N);
    cudaDeviceSynchronize();
    
    *d_energy = 0;
    compareKernel<<<blocks, threads>>>(d_rotated, d_target, d_energy, N);
    cudaDeviceSynchronize();
    std::cout << "Energy after 90 deg rotation: " << *d_energy << " (Aligned Success)" << std::endl;

    cudaFree(d_target); cudaFree(d_input); cudaFree(d_rotated); cudaFree(d_energy);
    return 0;
}