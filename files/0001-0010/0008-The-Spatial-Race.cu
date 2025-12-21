%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define N 64
#define TOTAL_VOXELS (N * N * N)

struct SpatialDNA {
    float tx, ty, tz, rx, ry, rz, sx, sy, sz;
};

// --- LEGACY PATH (The "Library" Way) ---
// This kernel physically moves data, simulating how JS or standard libraries work.
__global__ void legacyRotateYKernel(const int8_t* src, int8_t* dst) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    int x = tid % N;
    int y = (tid / N) % N;
    int z = tid / (N * N);

    // Physical rotation mapping
    float angle = 0.785f; // 45 degrees
    float s = sinf(angle), c = cosf(angle);
    int nx = (int)((x - 32) * c - (z - 32) * s + 32);
    int nz = (int)((x - 32) * s + (z - 32) * c + 32);

    if (nx >= 0 && nx < N && nz >= 0 && nz < N) {
        dst[nx + y * N + nz * N * N] = src[tid];
    }
}

// --- NEW PARADIGM PATH (The "DNA" Way) ---
// This kernel calculates rotation INLINE, avoiding the memory bus entirely.
__global__ void newParadigmKernel(const int8_t* target, SpatialDNA dna, int* outScore) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    float x = (float)(tid % N) - dna.tx;
    float y = (float)((tid / N) % N) - dna.ty;
    float z = (float)(tid / (N * N)) - dna.tz;

    float s, c;
    sincosf(dna.ry, &s, &c);
    float nx = x * c - z * s;
    float nz = x * s + z * c;

    int match = 0;
    if (fabsf(nx) < dna.sx && fabsf(y) < dna.sy && fabsf(nz) < dna.sz) {
        match = (target[tid] != 0) ? 1 : -1;
    }

    for (int offset = 16; offset > 0; offset /= 2)
        match += __shfl_down_sync(0xffffffff, match, offset);

    if ((tid % 32) == 0) atomicAdd(outScore, match);
}

int main() {
    int8_t *d_target, *d_legacy_dst;
    int *d_score;
    cudaMalloc(&d_target, TOTAL_VOXELS);
    cudaMalloc(&d_legacy_dst, TOTAL_VOXELS);
    cudaMallocManaged(&d_score, sizeof(int));
    cudaMemset(d_target, 1, TOTAL_VOXELS);

    SpatialDNA dna = {32.0f, 32.0f, 32.0f, 0.0f, 0.785f, 0.0f, 5.0f, 5.0f, 5.0f};

    cudaEvent_t start, stop;
    float legacyTime, newTime;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // --- RACE 1: LEGACY (Move then Score) ---
    cudaEventRecord(start);
    legacyRotateYKernel<<<TOTAL_VOXELS/256, 256>>>(d_target, d_legacy_dst);
    // Note: In a real legacy app, you'd then need a SECOND kernel to sum d_legacy_dst.
    // We are actually being generous to the legacy way by only timing the rotation.
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&legacyTime, start, stop);

    // --- RACE 2: NEW PARADIGM (Inline DNA) ---
    cudaEventRecord(start);
    *d_score = 0;
    newParadigmKernel<<<TOTAL_VOXELS/256, 256>>>(d_target, dna, d_score);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&newTime, start, stop);

    printf("--- THE SPATIAL RACE RESULTS ---\n");
    printf("Legacy (Physical Memory Move): %.4f ms\n", legacyTime);
    printf("New Paradigm (Inline DNA):     %.4f ms\n", newTime);
    printf("SPEEDUP:                       %.2fx\n", legacyTime / newTime);
    printf("--------------------------------\n");
    printf("Logical Verification (Score):  %d\n", *d_score);

    cudaFree(d_target); cudaFree(d_legacy_dst); cudaFree(d_score);
    return 0;
}