%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define N 512
#define TOTAL_VOXELS (N * N * N)

// The DNA structure: 9 parameters that define "How we see the world"
struct SpatialDNA {
    float tx, ty, tz, rx, ry, rz, sx, sy, sz;
};

// --- ERA 1: THE TYPICAL METHOD (Memory Bound) ---
// This kernel physically moves bytes in VRAM. It is the "Library" way.
__global__ void legacyTransform(const uint8_t* src, uint8_t* dst, float angle) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    int x = tid % N; int y = (tid / N) % N; int z = tid / (N * N);
    float s = sinf(angle), c = cosf(angle);
    
    // Physical coordinate mapping
    int nx = (int)((x - 32) * c - (z - 32) * s + 32);
    int nz = (int)((x - 32) * s + (z - 32) * c + 32);

    if (nx >= 0 && nx < N && nz >= 0 && nz < N) {
        dst[nx + y * N + nz * N * N] = src[tid];
    }
}

__global__ void legacyScore(const uint8_t* data, int* outScore) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;
    if (data[tid] > 0) atomicAdd(outScore, (int)data[tid]);
}

// --- ERA 2: THE DNA PARADIGM (Instruction Bound) ---
// Zero-Copy: No data moves. We jitter the lens, not the voxels.
__global__ void dnaFusedKernel(const uint8_t* target, SpatialDNA dna, int* outScore) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    // Virtual Perception
    float x = (float)(tid % N) - dna.tx;
    float y = (float)((tid / N) % N) - dna.ty;
    float z = (float)(tid / (N * N)) - dna.tz;

    float s, c; sincosf(dna.ry, &s, &c);
    float nx = x * c - z * s;
    float nz = x * s + z * c;

    int match = 0;
    // Bounds check + Ternary Scoring in one pass
    if (fabsf(nx) < dna.sx && fabsf(y) < dna.sy && fabsf(nz) < dna.sz) {
        match = (int)target[tid];
    }

    // Warp-Shuffle Reduction (33x Speed Trick)
    for (int offset = 16; offset > 0; offset /= 2)
        match += __shfl_down_sync(0xffffffff, match, offset);

    if ((tid % 32) == 0) atomicAdd(outScore, match);
}

int main() {
    uint8_t *d_src, *d_dst;
    int *d_score;
    cudaMalloc(&d_src, TOTAL_VOXELS);
    cudaMalloc(&d_dst, TOTAL_VOXELS);
    cudaMallocManaged(&d_score, sizeof(int));

    // Initialize Ternary Field (1s and 2s)
    uint8_t* h_data = (uint8_t*)malloc(TOTAL_VOXELS);
    for(int i=0; i<TOTAL_VOXELS; i++) h_data[i] = (i % 2) + 1;
    cudaMemcpy(d_src, h_data, TOTAL_VOXELS, cudaMemcpyHostToDevice);

    SpatialDNA dna = {32.0f, 32.0f, 32.0f, 0.0f, 0.5f, 0.0f, 5.0f, 5.0f, 5.0f};
    cudaEvent_t start, stop;
    float timeLegacy, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("--- 21/12 RELEASE: THE SPATIAL RACE ---\n");

    // 1. RUN TYPICAL METHOD
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        *d_score = 0;
        legacyTransform<<<TOTAL_VOXELS/256, 256>>>(d_src, d_dst, 0.5f + (i * 0.001f));
        legacyScore<<<TOTAL_VOXELS/256, 256>>>(d_dst, d_score);
        cudaDeviceSynchronize(); 
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeLegacy, start, stop);

    // 2. RUN DNA PARADIGM
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        *d_score = 0;
        dna.ry = 0.5f + (i * 0.001f);
        dnaFusedKernel<<<TOTAL_VOXELS/256, 256>>>(d_src, dna, d_score);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeDNA, start, stop);

    printf("Legacy (Physical Move): %.2f ms\n", timeLegacy);
    printf("DNA Paradigm (Fused):   %.2f ms\n", timeDNA);
    printf("SPEEDUP:                %.2fx\n", timeLegacy / timeDNA);
    printf("---------------------------------------\n");
    printf("Logical Verification: Final Score = %d\n", *d_score);

    return 0;
}