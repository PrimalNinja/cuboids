#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

#define N 64
#define BATCH_SIZE 64
#define TOTAL_VOXELS (N * N * N)
#define GEN_COUNT 500

struct SpatialDNA {
    float tx, ty, tz; 
    float ry; 
};

// --- TRACK A: TERNARY DNA INFERENCE ---
__global__ void dnaAdaptiveKernel(int8_t* target, int8_t* brains, SpatialDNA* spatial, int* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    float lx = (float)(tid % N) - (N/2);
    float ly = (float)((tid / N) % N) - (N/2);
    float lz = (float)(tid / (N * N)) - (N/2);

    float s = __sinf(spatial[b].ry);
    float c = __cosf(spatial[b].ry);

    float rx = lx * c - lz * s;
    float rz = lx * s + lz * c;

    int gx = (int)(rx + spatial[b].tx + (N/2));
    int gy = (int)(ly + spatial[b].ty + (N/2));
    int gz = (int)(rz + spatial[b].tz + (N/2));

    int8_t match = 0;
    if (gx >= 0 && gx < N && gy >= 0 && gy < N && gz >= 0 && gz < N) {
        int targetIdx = gx + gy * N + gz * N * N;
        if (target[targetIdx] != 0 && brains[b * TOTAL_VOXELS + tid] != 0)
            match = (target[targetIdx] == brains[b * TOTAL_VOXELS + tid]) ? 1 : -1;
    }

    __shared__ int cache[256];
    cache[threadIdx.x] = (int)match;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&energies[b], cache[0]);
}

// --- TRACK B: TRADITIONAL FP32 INFERENCE ---
__global__ void tradAdaptiveKernel(float* target, float* brains, SpatialDNA* spatial, float* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    float lx = (float)(tid % N) - (N/2);
    float ly = (float)((tid / N) % N) - (N/2);
    float lz = (float)(tid / (N * N)) - (N/2);

    float s = __sinf(spatial[b].ry);
    float c = __cosf(spatial[b].ry);

    float rx = lx * c - lz * s;
    float rz = lx * s + lz * c;

    int gx = (int)(rx + spatial[b].tx + (N/2));
    int gy = (int)(ly + spatial[b].ty + (N/2));
    int gz = (int)(rz + spatial[b].tz + (N/2));

    float score = 0;
    if (gx >= 0 && gx < N && gy >= 0 && gy < N && gz >= 0 && gz < N) {
        score = target[gx + gy * N + gz * N * N] * brains[b * TOTAL_VOXELS + tid];
    }

    __shared__ float fCache[256];
    fCache[threadIdx.x] = score;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) fCache[threadIdx.x] += fCache[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&energies[b], fCache[0]);
}

__global__ void adaptiveMutateKernel(int8_t* bDNA, float* bTrad, SpatialDNA* spatial, int winnerIdx, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    int8_t champDNA = bDNA[winnerIdx * TOTAL_VOXELS + tid];
    float champTrad = bTrad[winnerIdx * TOTAL_VOXELS + tid];
    SpatialDNA champPos = spatial[winnerIdx];

    curandState state;
    curand_init(seed, tid, 0, &state);

    for (int b = 0; b < BATCH_SIZE; b++) {
        if (b == winnerIdx) continue;
        
        // Voxel Mutation
        if (curand_uniform(&state) < 0.05f) {
            bDNA[b * TOTAL_VOXELS + tid] = (int8_t)((curand_uniform(&state) * 3) - 1);
            bTrad[b * TOTAL_VOXELS + tid] = curand_uniform(&state) * 2.0f - 1.0f;
        } else {
            bDNA[b * TOTAL_VOXELS + tid] = champDNA;
            bTrad[b * TOTAL_VOXELS + tid] = champTrad;
        }

        // Spatial DNA Mutation (Bifurcated: Even = Precise, Odd = Wild)
        if (tid == 0) {
            float strength = (b % 2 == 0) ? 0.1f : 1.0f;
            spatial[b].tx = champPos.tx + (curand_uniform(&state) - 0.5f) * 2.0f * strength;
            spatial[b].ty = champPos.ty + (curand_uniform(&state) - 0.5f) * 2.0f * strength;
            spatial[b].tz = champPos.tz + (curand_uniform(&state) - 0.5f) * 2.0f * strength;
            spatial[b].ry = champPos.ry + (curand_uniform(&state) - 0.5f) * 0.5f * strength; 
        }
    }
}

int main() {
    int8_t *d_tDNA, *d_bDNA; int *d_eDNA;
    float *d_tTrad, *d_bTrad, *d_eTrad;
    SpatialDNA *d_spatial;
    int *d_winner;

    cudaMallocManaged(&d_tDNA, TOTAL_VOXELS);
    cudaMallocManaged(&d_bDNA, BATCH_SIZE * TOTAL_VOXELS);
    cudaMallocManaged(&d_eDNA, BATCH_SIZE * sizeof(int));
    cudaMallocManaged(&d_tTrad, TOTAL_VOXELS * sizeof(float));
    cudaMallocManaged(&d_bTrad, BATCH_SIZE * TOTAL_VOXELS * sizeof(float));
    cudaMallocManaged(&d_eTrad, BATCH_SIZE * sizeof(float));
    cudaMallocManaged(&d_spatial, BATCH_SIZE * sizeof(SpatialDNA));
    cudaMallocManaged(&d_winner, sizeof(int));

    // Initialize Target (Diagonal Line)
    cudaMemset(d_tDNA, 0, TOTAL_VOXELS);
    for(int i=0; i<20; i++) {
        int c = 32 + (i - 10);
        d_tDNA[c + 32*N + c*N*N] = 1;
        d_tTrad[c + 32*N + c*N*N] = 1.0f;
    }

    cudaEvent_t s1, e1, s2, e2;
    cudaEventCreate(&s1); cudaEventCreate(&e1);
    cudaEventCreate(&s2); cudaEventCreate(&e2);

    // RACE START: TERNARY DNA
    cudaEventRecord(s1);
    for (int g = 0; g < GEN_COUNT; g++) {
        cudaMemset(d_eDNA, 0, BATCH_SIZE * sizeof(int));
        dnaAdaptiveKernel<<<dim3(TOTAL_VOXELS/256+1, BATCH_SIZE), 256>>>(d_tDNA, d_bDNA, d_spatial, d_eDNA);
        cudaDeviceSynchronize();
        int bestIdx = 0, bestVal = -9999;
        for(int i=0; i<BATCH_SIZE; i++) if(d_eDNA[i] > bestVal) { bestVal = d_eDNA[i]; bestIdx = i; }
        *d_winner = bestIdx;
        adaptiveMutateKernel<<<TOTAL_VOXELS/256+1, 256>>>(d_bDNA, d_bTrad, d_spatial, *d_winner, 7979+g);
    }
    cudaEventRecord(e1);

    // RACE START: TRADITIONAL FP32
    cudaEventRecord(s2);
    for (int g = 0; g < GEN_COUNT; g++) {
        cudaMemset(d_eTrad, 0, BATCH_SIZE * sizeof(float));
        tradAdaptiveKernel<<<dim3(TOTAL_VOXELS/256+1, BATCH_SIZE), 256>>>(d_tTrad, d_bTrad, d_spatial, d_eTrad);
    }
    cudaEventRecord(e2);

    cudaEventSynchronize(e1); cudaEventSynchronize(e2);
    float tDNA, tTrad;
    cudaEventElapsedTime(&tDNA, s1, e1);
    cudaEventElapsedTime(&tTrad, s2, e2);

    std::cout << "--- 0079 ADAPTIVE BENCHMARK ---" << std::endl;
    std::cout << "Ternary Throughput: " << (double)TOTAL_VOXELS * BATCH_SIZE * GEN_COUNT / (tDNA/1000.0) / 1e9 << " GVox/s" << std::endl;
    std::cout << "Traditional Throughput: " << (double)TOTAL_VOXELS * BATCH_SIZE * GEN_COUNT / (tTrad/1000.0) / 1e9 << " GVox/s" << std::endl;
    std::cout << "Speedup Factor: " << tTrad / tDNA << "x" << std::endl;
    std::cout << "Winning Rotation: " << d_spatial[*d_winner].ry << " rad" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    return 0;
}