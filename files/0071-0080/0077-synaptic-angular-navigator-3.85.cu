%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define N 64
#define BATCH_SIZE 64
#define TOTAL_VOXELS (N * N * N)
#define GEN_COUNT 300

struct SpatialDNA { float tx, ty, tz, ry; };

__global__ void initRNG(curandState* state) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < TOTAL_VOXELS) curand_init(1234, tid, 0, &state[tid]);
}

// --- DNA KERNEL: VECTORIZED 4-VOXEL READS ---
__global__ void dnaAngularKernel(int8_t* target, int8_t* brains, SpatialDNA* spatial, int* energies) {
    int b = blockIdx.y;
    // Each thread processes 4 voxels to amortize the cost of trig/rotation math
    int tidBase = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (tidBase >= TOTAL_VOXELS) return;

    __shared__ float s, c, tx, ty, tz;
    if (threadIdx.x == 0) {
        sincosf(spatial[b].ry, &s, &c);
        tx = spatial[b].tx; ty = spatial[b].ty; tz = spatial[b].tz;
    }
    __syncthreads();

    int matchAccumulator = 0;
    for(int i = 0; i < 4; i++) {
        int tid = tidBase + i;
        float lx = (float)(tid % N) - (N/2);
        float ly = (float)((tid / N) % N) - (N/2);
        float lz = (float)(tid / (N * N)) - (N/2);

        int gx = (int)(lx * c - lz * s + tx + (N/2));
        int gy = (int)(ly + ty + (N/2));
        int gz = (int)(lx * s + lz * c + tz + (N/2));

        if (gx >= 0 && gx < N && gy >= 0 && gy < N && gz >= 0 && gz < N) {
            matchAccumulator += (target[gx + gy*N + gz*N*N] == brains[b * TOTAL_VOXELS + tid]) ? 1 : -1;
        }
    }

    __shared__ int cache[256];
    cache[threadIdx.x] = matchAccumulator;
    __syncthreads();
    for (int i = 128; i > 0; i >>= 1) {
        if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&energies[b], cache[0]);
}

// --- TRAD KERNEL: 1-VOXEL PER THREAD ---
__global__ void tradAngularKernel(float* target, float* brains, SpatialDNA* spatial, float* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    __shared__ float s, c, tx, ty, tz;
    if (threadIdx.x == 0) {
        sincosf(spatial[b].ry, &s, &c);
        tx = spatial[b].tx; ty = spatial[b].ty; tz = spatial[b].tz;
    }
    __syncthreads();

    float lx = (float)(tid % N) - (N/2);
    float ly = (float)((tid / N) % N) - (N/2);
    float lz = (float)(tid / (N * N)) - (N/2);

    int gx = (int)(lx * c - lz * s + tx + (N/2));
    int gy = (int)(ly + ty + (N/2));
    int gz = (int)(lx * s + lz * c + tz + (N/2));

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

template<typename T>
__global__ void angularMutate(T* brains, SpatialDNA* spatial, void* energies, int* winnerIdx, curandState* rng, bool isFloat) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;
    __shared__ int champ;
    if (threadIdx.x == 0) {
        float bestVal = -2e9; int bestIdx = 0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            float val = isFloat ? ((float*)energies)[i] : (float)((int*)energies)[i];
            if (val > bestVal) { bestVal = val; bestIdx = i; }
        }
        champ = bestIdx; *winnerIdx = bestIdx;
    }
    __syncthreads();
    curandState localRNG = rng[tid];
    T cGene = brains[champ * TOTAL_VOXELS + tid];
    for (int b = 0; b < BATCH_SIZE; b++) {
        if (b == champ) continue;
        if (curand_uniform(&localRNG) < 0.05f) {
            if constexpr (std::is_same_v<T, int8_t>) brains[b * TOTAL_VOXELS + tid] = (int8_t)((curand_uniform(&localRNG)*3)-1);
            else brains[b * TOTAL_VOXELS + tid] = (curand_uniform(&localRNG)*2.0f)-1.0f;
        } else brains[b * TOTAL_VOXELS + tid] = cGene;
        if (tid == 0) {
            spatial[b].tx = spatial[champ].tx + (curand_uniform(&localRNG)-0.5f)*2.0f;
            spatial[b].ty = spatial[champ].ty + (curand_uniform(&localRNG)-0.5f)*2.0f;
            spatial[b].tz = spatial[champ].tz + (curand_uniform(&localRNG)-0.5f)*2.0f;
            spatial[b].ry = spatial[champ].ry + (curand_uniform(&localRNG)-0.5f)*0.3f;
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

    for(int i=0; i<TOTAL_VOXELS; i++) { d_tDNA[i] = 1; d_tTrad[i] = 1.0f; }
    initRNG<<<TOTAL_VOXELS/256+1, 256>>>(d_rng);
    cudaDeviceSynchronize();

    cudaEvent_t s1, e1, s2, e2;
    cudaEventCreate(&s1); cudaEventCreate(&e1); cudaEventCreate(&s2); cudaEventCreate(&e2);

    cudaEventRecord(s1);
    for (int g = 0; g < GEN_COUNT; g++) {
        cudaMemset(d_eDNA, 0, BATCH_SIZE * sizeof(int));
        // Launch DNA with 1/4th the threads because each thread handles 4 voxels
        dnaAngularKernel<<<(TOTAL_VOXELS/4)/256+1, dim3(256, BATCH_SIZE)>>>(d_tDNA, d_bDNA, d_sDNA, d_eDNA);
        angularMutate<int8_t><<<TOTAL_VOXELS/256+1, 256>>>(d_bDNA, d_sDNA, d_eDNA, d_winDNA, d_rng, false);
    }
    cudaEventRecord(e1);

    cudaEventRecord(s2);
    for (int g = 0; g < GEN_COUNT; g++) {
        cudaMemset(d_eTrad, 0, BATCH_SIZE * sizeof(float));
        tradAngularKernel<<<dim3(TOTAL_VOXELS/256+1, BATCH_SIZE), 256>>>(d_tTrad, d_bTrad, d_sTrad, d_eTrad);
        angularMutate<float><<<TOTAL_VOXELS/256+1, 256>>>(d_bTrad, d_sTrad, d_eTrad, d_winTrad, d_rng, true);
    }
    cudaEventRecord(e2);

    cudaDeviceSynchronize();
    float tDNA, tTrad;
    cudaEventElapsedTime(&tDNA, s1, e1); cudaEventElapsedTime(&tTrad, s2, e2);
    long long total_work = (long long)TOTAL_VOXELS * BATCH_SIZE * GEN_COUNT;

    std::cout << "--- SOVEREIGN ANGULAR RACE ---" << std::endl;
    std::cout << "DNA:  " << tDNA << " ms | " << (total_work / (tDNA/1000.0) / 1e9) << " GVox/s" << std::endl;
    std::cout << "Trad: " << tTrad << " ms | " << (total_work / (tTrad/1000.0) / 1e9) << " GVox/s" << std::endl;
    std::cout << "Speedup: " << tTrad/tDNA << "x" << std::endl;

    return 0;
}