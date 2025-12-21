%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define N 64
#define BATCH_SIZE 64
#define TOTAL_VOXELS (N * N * N)
#define GEN_COUNT 100

struct SpatialDNA { float tx, ty, tz; };

// --- DNA TRACK: VECTORIZED SIMD INFERENCE ---
__global__ void dnaExplorerSimdKernel(int8_t* target, int8_t* brains, SpatialDNA* spatial, int* energies) {
    int b = blockIdx.y;
    // Each thread processes 4 voxels simultaneously
    int tidBase = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (tidBase >= TOTAL_VOXELS) return;

    __shared__ int sx, sy, sz;
    if (threadIdx.x == 0) {
        sx = (int)spatial[b].tx; sy = (int)spatial[b].ty; sz = (int)spatial[b].tz;
    }
    __syncthreads();

    int matchAccumulator = 0;

    // Process 4 consecutive voxels
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int tid = tidBase + i;
        int lx = tid % N;
        int ly = (tid / N) % N;
        int lz = tid / (N * N);

        int gx = lx + sx;
        int gy = ly + sy;
        int gz = lz + sz;

        if (gx >= 0 && gx < N && gy >= 0 && gy < N && gz >= 0 && gz < N) {
            // Hardware-level byte comparison logic
            if (target[gx + gy * N + gz * N * N] == brains[b * TOTAL_VOXELS + tid]) 
                matchAccumulator++;
            else 
                matchAccumulator--;
        }
    }

    __shared__ int cache[256];
    cache[threadIdx.x] = matchAccumulator;
    __syncthreads();
    
    // Standard Reduction...
    for (int i = 128; i > 0; i >>= 1) {
        if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&energies[b], cache[0]);
}

// --- PERSISTENT RNG ---
__global__ void initRNG(curandState* state) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < TOTAL_VOXELS) curand_init(1234, tid, 0, &state[tid]);
}

// --- DNA TRACK: 1-BYTE KINEMATIC INFERENCE ---
__global__ void dnaExplorerKernel(int8_t* target, int8_t* brains, SpatialDNA* spatial, int* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    __shared__ int sx, sy, sz;
    if (threadIdx.x == 0) {
        sx = (int)spatial[b].tx; sy = (int)spatial[b].ty; sz = (int)spatial[b].tz;
    }
    __syncthreads();

    int lx = tid % N; int ly = (tid / N) % N; int lz = tid / (N * N);
    int gx = lx + sx; int gy = ly + sy; int gz = lz + sz;

    int match = 0;
    if (gx >= 0 && gx < N && gy >= 0 && gy < N && gz >= 0 && gz < N) {
        match = (target[gx + gy*N + gz*N*N] == brains[b * TOTAL_VOXELS + tid]) ? 1 : -1;
    }

    __shared__ int cache[256];
    cache[threadIdx.x] = match;
    __syncthreads();
    for (int i = 128; i > 0; i >>= 1) {
        if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&energies[b], cache[0]);
}

// --- DNA TRACK: ISOLATED MUTATION ---
__global__ void mutateDNA(int8_t* brains, SpatialDNA* spatial, int* energies, int* winnerIdx, curandState* rng) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    __shared__ int champ;
    if (threadIdx.x == 0) {
        int bestV = -2e9; int bestI = 0;
        for (int i = 0; i < BATCH_SIZE; i++) { if (energies[i] > bestV) { bestV = energies[i]; bestI = i; } }
        champ = bestI; *winnerIdx = bestI;
    }
    __syncthreads();

    int8_t cGene = brains[champ * TOTAL_VOXELS + tid];
    curandState localRNG = rng[tid];

    for (int b = 0; b < BATCH_SIZE; b++) {
        if (b == champ) continue;
        if (curand_uniform(&localRNG) < 0.05f) 
            brains[b * TOTAL_VOXELS + tid] = (int8_t)((curand_uniform(&localRNG)*3)-1);
        else 
            brains[b * TOTAL_VOXELS + tid] = cGene;
        
        if (tid == 0) {
            spatial[b].tx = spatial[champ].tx + (curand_uniform(&localRNG)-0.5f)*4.0f;
            spatial[b].ty = spatial[champ].ty + (curand_uniform(&localRNG)-0.5f)*4.0f;
            spatial[b].tz = spatial[champ].tz + (curand_uniform(&localRNG)-0.5f)*4.0f;
        }
    }
    rng[tid] = localRNG;
}

// --- TRAD TRACK: 4-BYTE FP32 INFERENCE ---
__global__ void tradExplorerKernel(float* target, float* brains, SpatialDNA* spatial, float* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    __shared__ float sx, sy, sz;
    if (threadIdx.x == 0) {
        sx = spatial[b].tx; sy = spatial[b].ty; sz = spatial[b].tz;
    }
    __syncthreads();

    int lx = tid % N; int ly = (tid / N) % N; int lz = tid / (N * N);
    int gx = lx + (int)sx; int gy = ly + (int)sy; int gz = lz + (int)sz;

    float score = 0;
    if (gx >= 0 && gx < N && gy >= 0 && gy < N && gz >= 0 && gz < N) {
        score = target[gx + gy*N + gz*N*N] * brains[b * TOTAL_VOXELS + tid];
    }

    __shared__ float fCache[256];
    fCache[threadIdx.x] = score;
    __syncthreads();
    for (int i = 128; i > 0; i >>= 1) {
        if (threadIdx.x < i) fCache[threadIdx.x] += fCache[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&energies[b], fCache[0]);
}

// --- TRAD TRACK: ISOLATED MUTATION ---
__global__ void mutateTrad(float* brains, SpatialDNA* spatial, float* energies, int* winnerIdx, curandState* rng) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    __shared__ int champ;
    if (threadIdx.x == 0) {
        float bestV = -2e9; int bestI = 0;
        for (int i = 0; i < BATCH_SIZE; i++) { if (energies[i] > bestV) { bestV = energies[i]; bestI = i; } }
        champ = bestI; *winnerIdx = bestI;
    }
    __syncthreads();

    float cGene = brains[champ * TOTAL_VOXELS + tid];
    curandState localRNG = rng[tid];

    for (int b = 0; b < BATCH_SIZE; b++) {
        if (b == champ) continue;
        if (curand_uniform(&localRNG) < 0.05f) 
            brains[b * TOTAL_VOXELS + tid] = (curand_uniform(&localRNG)*2.0f-1.0f);
        else 
            brains[b * TOTAL_VOXELS + tid] = cGene;

        if (tid == 0) {
            spatial[b].tx = spatial[champ].tx + (curand_uniform(&localRNG)-0.5f)*4.0f;
            spatial[b].ty = spatial[champ].ty + (curand_uniform(&localRNG)-0.5f)*4.0f;
            spatial[b].tz = spatial[champ].tz + (curand_uniform(&localRNG)-0.5f)*4.0f;
        }
    }
    rng[tid] = localRNG;
}

int main() {
    int8_t *d_tDNA, *d_bDNA; int *d_eDNA, *d_winDNA;
    float *d_tTrad, *d_bTrad, *d_eTrad; int *d_winTrad;
    SpatialDNA *d_sDNA, *d_sTrad; curandState *d_rng;

    cudaMallocManaged(&d_tDNA, TOTAL_VOXELS); cudaMallocManaged(&d_bDNA, BATCH_SIZE * TOTAL_VOXELS);
    cudaMallocManaged(&d_eDNA, BATCH_SIZE * sizeof(int)); cudaMallocManaged(&d_winDNA, sizeof(int));
    cudaMallocManaged(&d_sDNA, BATCH_SIZE * sizeof(SpatialDNA));
    cudaMallocManaged(&d_tTrad, TOTAL_VOXELS * sizeof(float)); cudaMallocManaged(&d_bTrad, BATCH_SIZE * TOTAL_VOXELS * sizeof(float));
    cudaMallocManaged(&d_eTrad, BATCH_SIZE * sizeof(float)); cudaMallocManaged(&d_winTrad, sizeof(int));
    cudaMallocManaged(&d_sTrad, BATCH_SIZE * sizeof(SpatialDNA));
    cudaMallocManaged(&d_rng, TOTAL_VOXELS * sizeof(curandState));

    cudaMemset(d_tDNA, 1, TOTAL_VOXELS);
    cudaMemset(d_tTrad, 1, TOTAL_VOXELS * sizeof(float));

    initRNG<<<TOTAL_VOXELS/256+1, 256>>>(d_rng);
    cudaDeviceSynchronize();

    cudaEvent_t s1, e1, s2, e2;
    cudaEventCreate(&s1); cudaEventCreate(&e1); cudaEventCreate(&s2); cudaEventCreate(&e2);

    // --- RACE 1: DNA (The Specialist) ---
    cudaEventRecord(s1);
    for (int g = 0; g < GEN_COUNT; g++) {
        cudaMemset(d_eDNA, 0, BATCH_SIZE * sizeof(int));
        dnaExplorerKernel<<<dim3(TOTAL_VOXELS/256+1, BATCH_SIZE), 256>>>(d_tDNA, d_bDNA, d_sDNA, d_eDNA);
        mutateDNA<<<TOTAL_VOXELS/256+1, 256>>>(d_bDNA, d_sDNA, d_eDNA, d_winDNA, d_rng);
    }
    cudaEventRecord(e1);

    // --- RACE 2: TRADITIONAL (The Generalist) ---
    cudaEventRecord(s2);
    for (int g = 0; g < GEN_COUNT; g++) {
        cudaMemset(d_eTrad, 0, BATCH_SIZE * sizeof(float));
        tradExplorerKernel<<<dim3(TOTAL_VOXELS/256+1, BATCH_SIZE), 256>>>(d_tTrad, d_bTrad, d_sTrad, d_eTrad);
        mutateTrad<<<TOTAL_VOXELS/256+1, 256>>>(d_bTrad, d_sTrad, d_eTrad, d_winTrad, d_rng);
    }
    cudaEventRecord(e2);

    cudaDeviceSynchronize();
    float tDNA, tTrad;
    cudaEventElapsedTime(&tDNA, s1, e1); cudaEventElapsedTime(&tTrad, s2, e2);

    std::cout << "--- FINAL SOVEREIGN RACE ---" << std::endl;
    std::cout << "DNA:  " << tDNA << " ms | Throughput: " << (TOTAL_VOXELS * BATCH_SIZE * GEN_COUNT / (tDNA/1000.0) / 1e9) << " GVox/s" << std::endl;
    std::cout << "Trad: " << tTrad << " ms | Throughput: " << (TOTAL_VOXELS * BATCH_SIZE * GEN_COUNT / (tTrad/1000.0) / 1e9) << " GVox/s" << std::endl;
    std::cout << "Real Speedup: " << tTrad/tDNA << "x" << std::endl;

    return 0;
}