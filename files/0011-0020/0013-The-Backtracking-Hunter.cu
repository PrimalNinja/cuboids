%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define N 64
#define TOTAL_VOXELS (N * N * N)

struct SpatialDNA {
    float tx, ty, tz, rx, ry, rz, sx, sy, sz;
};

// --- THE PERCEPTION ENGINE ---
__global__ void evaluateDNA(const uint8_t* target, SpatialDNA dna, int* outScore) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    float x = (float)(tid % N) - dna.tx;
    float y = (float)((tid / N) % N) - dna.ty;
    float z = (float)(tid / (N * N)) - dna.tz;

    float s, c; sincosf(dna.ry, &s, &c);
    float nx = x * c - z * s;
    float nz = x * s + z * c;

    int val = 0;
    if (fabsf(nx) < dna.sx && fabsf(y) < dna.sy && fabsf(nz) < dna.sz) {
        val = (int)target[tid];
    }

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if ((tid % 32) == 0) atomicAdd(outScore, val);
}

int main() {
    srand(time(NULL));
    uint8_t *d_data;
    int *d_score;
    cudaMalloc(&d_data, TOTAL_VOXELS);
    cudaMallocManaged(&d_score, sizeof(int));

    // Seed Ternary Field
    uint8_t* h_data = (uint8_t*)malloc(TOTAL_VOXELS);
    for(int i=0; i<TOTAL_VOXELS; i++) h_data[i] = (i % 2) + 1;
    cudaMemcpy(d_data, h_data, TOTAL_VOXELS, cudaMemcpyHostToDevice);

    // Initial State
    SpatialDNA bestDna = {32.0f, 32.0f, 32.0f, 0.0f, 0.0f, 0.0f, 5.0f, 5.0f, 5.0f};
    int targetScore = 1200; // Hunting for a specific density
    
    // Get Initial Score
    *d_score = 0;
    evaluateDNA<<<TOTAL_VOXELS/256, 256>>>(d_data, bestDna, d_score);
    cudaDeviceSynchronize();
    int bestScore = *d_score;
    int bestDiff = abs(bestScore - targetScore);

    printf("STARTING HUNT: Target = %d | Initial Score = %d\n", targetScore, bestScore);
    printf("--------------------------------------------------\n");

    // THE EVOLUTIONARY LOOP (Selection Pressure)
    for(int i = 0; i < 20; i++) {
        SpatialDNA trialDna = bestDna;
        
        // JITTER: Randomly mutate the rotation
        float mutation = ((float)rand()/(float)RAND_MAX - 0.5f) * 2.0f;
        trialDna.ry += mutation;

        // EVALUATE
        *d_score = 0;
        evaluateDNA<<<TOTAL_VOXELS/256, 256>>>(d_data, trialDna, d_score);
        cudaDeviceSynchronize();
        
        int trialScore = *d_score;
        int trialDiff = abs(trialScore - targetScore);

        if (trialDiff < bestDiff) {
            // EVOLUTIONARY SUCCESS: Keep the mutation
            bestDna = trialDna;
            bestScore = trialScore;
            bestDiff = trialDiff;
            printf("[KEEP] Iteration %d: New Score %d (Diff: %d) | RY: %.4f\n", i, bestScore, bestDiff, bestDna.ry);
        } else {
            // BACKTRACK: Discard the mutation
            // (We simply don't update bestDna)
        }
    }

    printf("--------------------------------------------------\n");
    printf("FINAL RESULT: Closest Score Found = %d\n", bestScore);

    cudaFree(d_data); cudaFree(d_score); free(h_data);
    return 0;
}