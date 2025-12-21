%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define N 512
#define TOTAL_VOXELS (N * N * N)

struct SpatialDNA {
    float tx, ty, tz, rx, ry, rz, sx, sy, sz;
};

// --- OPTIMIZED TRADITIONAL (Discrete Kernels) ---
__global__ void functionalRotate(const uint8_t* __restrict__ src, uint8_t* dst) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    int x = tid % N; int y = (tid / N) % N; int z = tid / (N * N);
    
    // Using hardware-accelerated math even for Traditional
    float s, c; __sincosf(0.785f, &s, &c); 
    
    int nx = (int)((x - 32) * c - (z - 32) * s + 32);
    int nz = (int)((x - 32) * s + (z - 32) * c + 32);
    
    if (nx >= 0 && nx < N && nz >= 0 && nz < N) 
        dst[nx + y * N + nz * N * N] = src[tid];
}

__global__ void functionalScore(const uint8_t* __restrict__ data, int* score) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;
    
    uint8_t val = data[tid];
    if (val > 0) atomicAdd(score, (int)val);
}

// --- OPTIMIZED DNA PARADIGM (Fused Logic) ---
__global__ void dnaArchitectKernel(const uint8_t* __restrict__ target, SpatialDNA dna, int* outScore) {
    __shared__ int blockTotal;
    if (threadIdx.x == 0) blockTotal = 0;
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    // Fast coordinate decomposition
    int z = tid / (N * N);
    int rem = tid % (N * N);
    int y = rem / N;
    int x = rem % N;

    float fx = (float)x - dna.tx;
    float fy = (float)y - dna.ty;
    float fz = (float)z - dna.tz;

    float s, c; __sincosf(dna.ry, &s, &c);
    float nx = fx * c - fz * s;
    float nz = fx * s + fz * c;

    int val = 0;
    // Spatial perception logic
    if (fabsf(nx) < dna.sx && fabsf(fy) < dna.sy && fabsf(nz) < dna.sz) {
        val = (int)target[tid];
    }

    // Warp Shuffle Reduction
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if ((threadIdx.x & 31) == 0) atomicAdd(&blockTotal, val);
    __syncthreads();

    if (threadIdx.x == 0 && blockTotal > 0) atomicAdd(outScore, blockTotal);
}

int main() {
    uint8_t *d_src, *d_dst;
    int *d_score;
    cudaMalloc(&d_src, TOTAL_VOXELS);
    cudaMalloc(&d_dst, TOTAL_VOXELS);
    cudaMallocManaged(&d_score, sizeof(int));
    cudaMemset(d_src, 1, TOTAL_VOXELS);

    SpatialDNA dna = {32.0f, 32.0f, 32.0f, 0.0f, 0.785f, 0.0f, 5.0f, 5.0f, 5.0f};
    cudaEvent_t start, stop;
    float timeTrad, timeFunc, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // 1. TYPICAL (Standard Sync Pipeline)
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        functionalRotate<<<TOTAL_VOXELS/256, 256>>>(d_src, d_dst);
        cudaDeviceSynchronize(); 
        functionalScore<<<TOTAL_VOXELS/256, 256>>>(d_dst, d_score);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTrad, start, stop);

    // 2. FUNCTIONAL GPU (Optimized Discrete Kernels)
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        functionalRotate<<<TOTAL_VOXELS/256, 256>>>(d_src, d_dst);
        functionalScore<<<TOTAL_VOXELS/256, 256>>>(d_dst, d_score);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeFunc, start, stop);

    // 3. DNA PARADIGM (Fully Fused Zero-Copy)
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        dnaArchitectKernel<<<TOTAL_VOXELS/256, 256>>>(d_src, dna, d_score);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeDNA, start, stop);

    printf("\n--- OPTIMIZED TRIPLE CROWN (100 Iterations) ---\n");
    printf("1. Typical (Sync Heavy):  %.3f ms\n", timeTrad);
    printf("2. Functional (Discrete): %.3f ms\n", timeFunc);
    printf("3. DNA Paradigm (Fused):  %.3f ms\n", timeDNA);
    printf("--------------------------------------------------\n");
    printf("Speedup (DNA vs Typical):    %.2fx\n", timeTrad / timeDNA);
    printf("Speedup (DNA vs Functional): %.2fx\n", timeFunc / timeDNA);

    return 0;
}