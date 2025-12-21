%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define IDX(x, y, z, n) ((x) * (n) * (n) + (y) * (n) + (z))

__global__ void rotateXKernel(const uint8_t* src, uint8_t* dst, int n, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int x = idx / (n * n);
        int y = (idx / n) % n;
        int z = idx % n;
        // X rotation logic
        int new_y = z;
        int new_z = n - 1 - y;
        dst[IDX(x, new_y, new_z, n)] = src[idx];
    }
}

int main() {
    printf("CUDA Source Compiled Successfully on Colab GPU.\n");
    return 0;
}