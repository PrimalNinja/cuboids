%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define IDX(x, y, z, n) ((x) * (n) * (n) + (y) * (n) + (z))
#define TERNARY_MOD 3

// --- PUBLIC HOOKS (The Kernels) ---

__global__ void rotateXKernel(const uint8_t* src, uint8_t* dst, int n, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int x = idx / (n * n), y = (idx / n) % n, z = idx % n;
    dst[IDX(x, z, n - 1 - y, n)] = src[idx];
}

__global__ void faceSumKernel(const uint8_t* face, int faceSize, uint32_t* outSum) {
    // Shared Memory is the "Fast Cache" for this specific Form instance
    extern __shared__ uint32_t sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < faceSize) ? (uint32_t)face[idx] : 0;
    __syncthreads();

    // Parallel Reduction Tree
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(outSum, sdata[0]);
}

// --- DISPATCHER (The Main Loop) ---

int main() {
    int n = 6;
    int total = n * n * n;
    int faceSize = n * n;
    size_t bytes = total * sizeof(uint8_t);

    // Host Setup
    uint8_t* h_data = (uint8_t*)malloc(bytes);
    for(int i=0; i<total; i++) h_data[i] = i % TERNARY_MOD; // Pattern for testing

    // Device Setup (The "Form" Memory)
    uint8_t *d_src, *d_dst;
    uint32_t *d_score, h_score = 0;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);
    cudaMalloc(&d_score, sizeof(uint32_t));
    
    cudaMemcpy(d_src, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_score, 0, sizeof(uint32_t));

    // TRIGGER HOOK 1: Action
    rotateXKernel<<<(total + 255) / 256, 256>>>(d_src, d_dst, n, total);
    
    // TRIGGER HOOK 2: Evaluation
    // Using 256 threads to sum up the first face of the cuboid
    faceSumKernel<<<1, 256, 256 * sizeof(uint32_t)>>>(d_dst, faceSize, d_score);

    // RETURN RESULT: Copy back to "Browser" (Host)
    cudaMemcpy(&h_score, d_score, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("AI Evaluation Hook Returned: %u\n", h_score);
    printf("Status: Full Cycle Complete (Rotate -> Score -> Return).\n");

    // Cleanup
    free(h_data);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_score);
    return 0;
}