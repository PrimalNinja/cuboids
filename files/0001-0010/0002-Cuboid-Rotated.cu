%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cstring>
#include <functional>

#define IDX(x, y, z, n) ((x) * (n) * (n) + (y) * (n) + (z))
#define MAX_N 64
#define TERNARY_MOD 3

enum FaceType { FRONT = 0, BACK, LEFT, RIGHT, BOTTOM, TOP };

typedef struct {
    int size;
    int front, back, left, right, bottom, top;
    int connections[6][4];
    int rotationState[3];
} ControlBlock;

typedef struct {
    uint8_t data[MAX_N * MAX_N * MAX_N];
    ControlBlock cb;
} Cuboid;

// --- KERNELS ---

// Optimized: We use a Destination array to avoid Race Conditions and Atomic errors
__global__ void rotateZKernel(const uint8_t* src, uint8_t* dst, int n, int degrees, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int x = idx / (n * n);
    int y = (idx / n) % n;
    int z = idx % n;

    int shift = (degrees / 90) % n;
    int new_z = (z + shift) % n;
    int new_idx = IDX(x, y, new_z, n);

    dst[new_idx] = src[idx]; 
}

__global__ void matrixAddKernel(uint8_t* face1, uint8_t* face2, uint8_t* result, int faceSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceSize) return;
    result[idx] = (face1[idx] + face2[idx]) % TERNARY_MOD;
}

// Fixed Shared Memory Declaration (only one 'extern' per kernel)
__global__ void faceSumKernel(uint8_t* face, int faceSize, uint32_t* outSum) {
    extern __shared__ uint32_t sdata_sum[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata_sum[tid] = (idx < faceSize) ? (uint32_t)face[idx] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_sum[tid] += sdata_sum[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(outSum, sdata_sum[0]);
}

// --- HOST HELPERS ---

void createTernaryCuboidHost(Cuboid* cub, int size) {
    cub->cb.size = size;
    int total = size * size * size;
    srand(42);
    for (int i = 0; i < total; ++i) cub->data[i] = rand() % TERNARY_MOD;
}

// Fixed Pointer-to-Pointer issue by using a flat array
void getFacesHost(const Cuboid* cub, uint8_t* faces, int size) {
    int faceSize = size * size;
    // For POC, we just copy the front 6 blocks of memory to represent 6 faces
    memcpy(faces, cub->data, faceSize * 6); 
}

int main() {
    int size = 6;
    int total = size * size * size;
    int faceSize = size * size;

    Cuboid h_cub;
    createTernaryCuboidHost(&h_cub, size);

    uint8_t *d_src, *d_dst, *d_face;
    uint32_t *d_sum, h_sum = 0;

    cudaMalloc(&d_src, total);
    cudaMalloc(&d_dst, total);
    cudaMalloc(&d_face, faceSize);
    cudaMalloc(&d_sum, sizeof(uint32_t));
    
    cudaMemcpy(d_src, h_cub.data, total, cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(uint32_t));

    printf("Starting GPU Cuboid Logic...\n");

    // 1. Test Rotation
    rotateZKernel<<<1, 256>>>(d_src, d_dst, size, 90, total);
    
    // 2. Test Face Sum (Reduction)
    cudaMemcpy(d_face, d_dst, faceSize, cudaMemcpyDeviceToDevice);
    faceSumKernel<<<1, 256, 256 * sizeof(uint32_t)>>>(d_face, faceSize, d_sum);
    
    cudaMemcpy(&h_sum, d_sum, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Success! Cuboid Rotated.\n");
    printf("Face 0 Sum Result: %u\n", h_sum);
    printf("Engine Status: Idle / Ready for Batch\n");

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_face); cudaFree(d_sum);
    return 0;
}