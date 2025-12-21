%%writefile cuboids_fixed.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define IDX(x, y, z, n) ((x) * (n) * (n) + (y) * (n) + (z))
#define N 6
#define TOTAL (N * N * N)
#define FACESIZE (N * N)

// True 90° Z-rotation (around vertical axis)
__global__ void rotateZKernelTrue(const uint8_t* src, uint8_t* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL) return;

    int x = idx / (n * n);
    int y = (idx / n) % n;
    int z = idx % n;
    
    // Rotate 90° clockwise around Z
    int new_x = n - 1 - y;  // x' = -y
    int new_y = x;          // y' = x
    int new_z = z;          // z unchanged
    
    dst[IDX(new_x, new_y, new_z, n)] = src[idx];
}

// Extract FRONT face (x=0 plane)
__global__ void extractFrontFace(const uint8_t* cube, uint8_t* face, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= FACESIZE) return;
    
    int y = idx / n;
    int z = idx % n;
    face[idx] = cube[IDX(0, y, z, n)];  // x=0 plane
}

// Face sum reduction
__global__ void faceSumKernel(uint8_t* face, uint32_t* result) {
    extern __shared__ uint32_t sdata[];
    int tid = threadIdx.x;
    
    sdata[tid] = (tid < FACESIZE) ? (uint32_t)face[tid] : 0;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) *result = sdata[0];
}

int main() {
    uint8_t h_cube[TOTAL];
    for(int i=0; i<TOTAL; i++) h_cube[i] = i % 3;
    
    uint8_t *d_cube, *d_rotated, *d_face;
    uint32_t *d_sum, h_sum;
    
    cudaMalloc(&d_cube, TOTAL);
    cudaMalloc(&d_rotated, TOTAL);
    cudaMalloc(&d_face, FACESIZE);
    cudaMalloc(&d_sum, sizeof(uint32_t));
    
    cudaMemcpy(d_cube, h_cube, TOTAL, cudaMemcpyHostToDevice);
    
    printf("GPU Cuboids v2 - Face Operations\n");
    
    // 1. True rotation
    rotateZKernelTrue<<<1, 256>>>(d_cube, d_rotated, N);
    
    // 2. Extract front face from rotated cube
    extractFrontFace<<<1, 36>>>(d_rotated, d_face, N);
    
    // 3. Sum face elements
    faceSumKernel<<<1, 36, 36*sizeof(uint32_t)>>>(d_face, d_sum);
    
    cudaMemcpy(&h_sum, d_sum, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    printf("Front face sum after 90° Z-rotation: %u\n", h_sum);
    printf("Expected: Sum of plane x=0 after rotating entire cube\n");
    
    cudaFree(d_cube); cudaFree(d_rotated); cudaFree(d_face); cudaFree(d_sum);
    return 0;
}