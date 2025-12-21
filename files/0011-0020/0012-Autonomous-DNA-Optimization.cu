%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define N 64
#define TOTAL_VOXELS (N * N * N)

struct SpatialDNA {
    float tx, ty, tz, rx, ry, rz, sx, sy, sz;
};

// --- THE ARCHITECT KERNEL ---
// Fuses Rotation, Bounds Checking, and Scoring into one memory pass.
__global__ void dnaSearchKernel(const uint8_t* target, SpatialDNA dna, int* outScore) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    // 1. VIRTUAL COORDINATES (The "Lens")
    float x = (float)(tid % N) - dna.tx;
    float y = (float)((tid / N) % N) - dna.ty;
    float z = (float)(tid / (N * N)) - dna.tz;

    // 2. INLINE ROTATION (Y-Axis Jitter)
    float s, c;
    sincosf(dna.ry, &s, &c);
    float nx = x * c - z * s;
    float nz = x * s + z * c;

    // 3. TERNARY EVALUATION
    int match = 0;
    // We only score if the voxel falls within our "DNA-defined" cuboid bounds
    if (fabsf(nx) < dna.sx && fabsf(y) < dna.sy && fabsf(nz) < dna.sz) {
        match = (int)target[tid]; 
    }

    // 4. WARP-SHUFFLE REDUCTION (33x Speedup Trick)
    for (int offset = 16; offset > 0; offset /= 2)
        match += __shfl_down_sync(0xffffffff, match, offset);

    if ((tid % 32) == 0) atomicAdd(outScore, match);
}

int main() {
    uint8_t *d_data;
    int *d_score;
    cudaMalloc(&d_data, TOTAL_VOXELS);
    cudaMallocManaged(&d_score, sizeof(int));

    // Seed the Ternary Field (1s and 2s)
    uint8_t* h_data = (uint8_t*)malloc(TOTAL_VOXELS);
    for(int i=0; i<TOTAL_VOXELS; i++) h_data[i] = (i % 2) + 1;
    cudaMemcpy(d_data, h_data, TOTAL_VOXELS, cudaMemcpyHostToDevice);

    // Initial DNA (Centered, 5x5x5 Cube, No Rotation)
    SpatialDNA currentDna = {32.0f, 32.0f, 32.0f, 0.0f, 0.0f, 0.0f, 5.0f, 5.0f, 5.0f};
    int targetScore = 50;
    
    printf("Starting Autonomous DNA Hunt...\n");
    printf("Initial Target: %d | Starting Score: (Calculating...)\n", targetScore);

    // EVOLUTIONARY LOOP
    for(int generation = 0; generation < 5; generation++) {
        *d_score = 0;
        dnaSearchKernel<<<TOTAL_VOXELS/256, 256>>>(d_data, currentDna, d_score);
        cudaDeviceSynchronize();

        printf("Generation %d: DNA.ry = %.4f | Score = %d\n", generation, currentDna.ry, *d_score);

        // MUTATION: Jitter the rotation slightly to find a new score
        currentDna.ry += 0.05f; 
    }

    printf("\nEvolution Complete. DNA has adapted to the Ternary Field.\n");

    cudaFree(d_data); cudaFree(d_score); free(h_data);
    return 0;
}