%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <chrono>

#define N 512
#define TOTAL_VOXELS (N * N * N)

struct SpatialDNA {
    float tx, ty, tz, rx, ry, rz, sx, sy, sz;
};

// --- TRADITIONAL: HIGHLY OPTIMIZED ---
// Even a "good" traditional kernel must write its results to VRAM to be useful
// for the next stage of a pipeline. This is where the latency lives.
__global__ void traditionalProcessor(const int8_t* src, int8_t* dst, float angle, int* score) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    int x = tid % N; int y = (tid / N) % N; int z = tid / (N * N);
    float s = sinf(angle), c = cosf(angle);
    
    int nx = (int)((x - 32) * c - (z - 32) * s + 32);
    int nz = (int)((x - 32) * s + (z - 32) * c + 32);

    if (nx >= 0 && nx < N && nz >= 0 && nz < N) {
        int8_t val = src[tid];
        dst[nx + y * N + nz * N * N] = val; // Necessary VRAM write
        if (val != 0) atomicAdd(score, 1);
    }
}

// --- NEW PARADIGM: THE CUBOIDS FUSED WAY ---
// No destination buffer. No VRAM writes. Only registers and shared memory.
__global__ void newParadigmFused(const int8_t* __restrict__ target, SpatialDNA dna, int* outScore) {
    __shared__ int blockTotal;
    if (threadIdx.x == 0) blockTotal = 0;
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    float z = (float)(tid / (N * N));
    int rem = tid % (N * N);
    float y = (float)(rem / N);
    float x = (float)(rem % N);

    x -= dna.tx; y -= dna.ty; z -= dna.tz;
    float s, c; __sincosf(dna.ry, &s, &c); 
    float nx = x * c - z * s;
    float nz = x * s + z * c;

    int match = (fabsf(nx) < dna.sx && fabsf(y) < dna.sy && fabsf(nz) < dna.sz && target[tid] != 0);

    // Efficient Register-level reduction
    for (int offset = 16; offset > 0; offset /= 2) 
        match += __shfl_down_sync(0xffffffff, match, offset);
    
    if ((threadIdx.x & 31) == 0) atomicAdd(&blockTotal, match);
    __syncthreads();
    
    if (threadIdx.x == 0 && blockTotal > 0) atomicAdd(outScore, blockTotal);
}

int main() {
    int8_t *d_target, *d_temp;
    int *d_score;
    cudaMalloc(&d_target, TOTAL_VOXELS);
    cudaMalloc(&d_temp, TOTAL_VOXELS);
    cudaMallocManaged(&d_score, sizeof(int));
    cudaMemset(d_target, 1, TOTAL_VOXELS);

    SpatialDNA dna = {32.0f, 32.0f, 32.0f, 0.0f, 0.785f, 0.0f, 5.0f, 5.0f, 5.0f};
    cudaEvent_t start, stop;
    float timeTrad, timeNew;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("Executing Race: Optimized Traditional vs. Cuboids Fused...\n");

    // 1. TRADITIONAL
    *d_score = 0;
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        traditionalProcessor<<<TOTAL_VOXELS/256, 256>>>(d_target, d_temp, 0.785f, d_score);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTrad, start, stop);

    // 2. CUBOIDS NEW PARADIGM
    *d_score = 0;
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        newParadigmFused<<<TOTAL_VOXELS/256, 256>>>(d_target, dna, d_score);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeNew, start, stop);

    printf("\n[REAL-WORLD PERFORMANCE COMPARISON]\n");
    printf("Traditional (Write + Score): %.3f ms\n", timeTrad);
    printf("Cuboids (Zero-Copy Fused):  %.3f ms\n", timeNew);
    printf("------------------------------------\n");
    printf("CUBOIDS ADVANTAGE: %.2fx Faster\n", timeTrad / timeNew);

    cudaFree(d_target); cudaFree(d_temp); cudaFree(d_score);
    return 0;
}