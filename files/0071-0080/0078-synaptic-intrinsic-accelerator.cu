%%writefile 0078-synaptic-intrinsic-accelerator.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

#define TEST_ID 37.02
#define N 64
#define BATCH_SIZE 64
#define TOTAL_VOXELS (N * N * N)
#define GEN_COUNT 300

struct SpatialDNA {
    float tx, ty, tz; 
    float ry; 
};

// --- DNA KERNEL: INTRINSIC ANGULAR INFERENCE ---
__global__ void dnaIntrinsicKernel(int8_t* target, int8_t* brains, SpatialDNA* spatial, int* energies) {
    int b = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    float lx = (float)(tid % N) - (N/2);
    float ly = (float)((tid / N) % N) - (N/2);
    float lz = (float)(tid / (N * N)) - (N/2);

    // Using Fast Intrinsics for maximum hardware throughput
    // These skip several CPU-style precision checks for raw speed
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
        if (target[targetIdx] != 0 && brains[b * TOTAL_VOXELS + tid] != 0) {
            match = (target[targetIdx] == brains[b * TOTAL_VOXELS + tid]) ? 1 : -1;
        }
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

__global__ void intrinsicMutateKernel(int8_t* brains, SpatialDNA* spatial, int winnerIdx, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;

    int8_t championGene = brains[winnerIdx * TOTAL_VOXELS + tid];
    SpatialDNA championPos = spatial[winnerIdx];
    curandState state;
    curand_init(seed, tid, 0, &state);

    for (int b = 0; b < BATCH_SIZE; b++) {
        if (b == winnerIdx) continue;
        
        // Voxel Mutation logic
        brains[b * TOTAL_VOXELS + tid] = (curand_uniform(&state) < 0.05f) ? (int8_t)((curand_uniform(&state) * 3) - 1) : championGene;

        if (tid == 0) {
            spatial[b].tx = championPos.tx + (curand_uniform(&state) - 0.5f) * 2.0f;
            spatial[b].ty = championPos.ty + (curand_uniform(&state) - 0.5f) * 2.0f;
            spatial[b].tz = championPos.tz + (curand_uniform(&state) - 0.5f) * 2.0f;
            spatial[b].ry = championPos.ry + (curand_uniform(&state) - 0.5f) * 0.4f; 
        }
    }
}

int main() {
    int8_t *d_target, *d_brains;
    int *d_energies, *d_winner;
    SpatialDNA *d_spatial;
    
    cudaMallocManaged(&d_target, TOTAL_VOXELS);
    cudaMallocManaged(&d_brains, BATCH_SIZE * TOTAL_VOXELS);
    cudaMallocManaged(&d_energies, BATCH_SIZE * sizeof(int));
    cudaMallocManaged(&d_winner, sizeof(int));
    cudaMallocManaged(&d_spatial, BATCH_SIZE * sizeof(SpatialDNA));

    cudaMemset(d_target, 0, TOTAL_VOXELS);
    for(int i=0; i<20; i++) {
        int coord = 32 + (i - 10);
        d_target[coord + 32*N + coord*N*N] = 1; 
    }

    for(int i=0; i<BATCH_SIZE; i++) d_spatial[i] = {0, 0, 0, 0};

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int g = 0; g < GEN_COUNT; g++) {
        cudaMemset(d_energies, 0, BATCH_SIZE * sizeof(int));
        dnaIntrinsicKernel<<<dim3(TOTAL_VOXELS/256+1, BATCH_SIZE), 256>>>(d_target, d_brains, d_spatial, d_energies);
        cudaDeviceSynchronize();

        int bestVal = -1000, bestIdx = 0;
        for(int i=0; i<BATCH_SIZE; i++) {
            if(d_energies[i] > bestVal) { bestVal = d_energies[i]; bestIdx = i; }
        }
        *d_winner = bestIdx;

        intrinsicMutateKernel<<<TOTAL_VOXELS/256+1, 256>>>(d_brains, d_spatial, *d_winner, 888+g);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    double total_ops = (double)TOTAL_VOXELS * BATCH_SIZE * GEN_COUNT;
    double throughput = (total_ops / (ms / 1000.0)) / 1e9;

    std::cout << "--- 0078 INTRINSIC ACCELERATOR PROFILE ---" << std::endl;
    std::cout << "TEST ID: " << TEST_ID << std::endl;
    std::cout << "Throughput: " << throughput << " GVox/s" << std::endl;
    std::cout << "Winning Rotation: " << d_spatial[*d_winner].ry << " rad" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    return 0;
}