%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define IDX(x, y, z, n) ((x) * (n) * (n) + (y) * (n) + (z))
#define TERNARY_MOD 3

__global__ void rotateXKernel(const uint8_t* src, uint8_t* dst, int n, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int x = idx / (n * n), y = (idx / n) % n, z = idx % n;
    // Map to a new spatial coordinate
    int new_y = z;
    int new_z = n - 1 - y;
    dst[IDX(x, new_y, new_z, n)] = src[idx];
}

__global__ void faceSumKernel(const uint8_t* data, int faceSize, uint32_t* outSum) {
    extern __shared__ uint32_t sdata[];
    int tid = threadIdx.x;
    
    // Each thread loads one voxel value from the first face
    sdata[tid] = (tid < faceSize) ? (uint32_t)data[tid] : 0;
    __syncthreads();

    // Tournament Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(outSum, sdata[0]);
}

int main() {
    int n = 6;
    int total = n * n * n;
    int faceSize = n * n; // 36 voxels

    uint8_t *h_data = (uint8_t*)malloc(total);
    // Fill with a dense ternary pattern (no zeros)
    for(int i=0; i<total; i++) h_data[i] = (i % 2) + 1; // Pattern of 1s and 2s

    uint8_t *d_src, *d_dst;
    uint32_t *d_score, h_score = 0;
    cudaMalloc(&d_src, total);
    cudaMalloc(&d_dst, total);
    cudaMalloc(&d_score, sizeof(uint32_t));
    
    cudaMemcpy(d_src, h_data, total, cudaMemcpyHostToDevice);
    cudaMemset(d_score, 0, sizeof(uint32_t));

    // Action
    rotateXKernel<<<(total + 255) / 256, 256>>>(d_src, d_dst, n, total);
    
    // Evaluation (Summing the first 36 elements of the result)
    faceSumKernel<<<1, 256, 256 * sizeof(uint32_t)>>>(d_dst, faceSize, d_score);

    cudaMemcpy(&h_score, d_score, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("AI Evaluation Hook Returned: %u\n", h_score);
    printf("Expected Sum (if dense): Between 36 and 72\n");
    
    free(h_data); cudaFree(d_src); cudaFree(d_dst); cudaFree(d_score);
    return 0;
}