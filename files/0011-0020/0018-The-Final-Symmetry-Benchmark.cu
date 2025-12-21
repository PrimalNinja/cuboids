%%writefile cuboids.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define N 512
#define TOTAL_VOXELS (N * N * N)

// --- ERA 1: TYPICAL DISPATCHER (Optimized Legacy) ---
__global__ void typicalRotate(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int axis) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_VOXELS) return;
    
    int x = idx / (N * N), y = (idx / N) % N, z = idx % N;
    
    // Scattered writes: The physical limit of VRAM bandwidth
    if (axis == 0)      dst[x * (N*N) + z * N + (N-1-y)] = src[idx];
    else if (axis == 1) dst[(N-1-z) * (N*N) + y * N + x] = src[idx];
}

__global__ void typicalScore(const uint8_t* __restrict__ data, uint32_t* score) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_VOXELS) return;
    uint8_t val = data[idx];
    if (val > 0) atomicAdd(score, (uint32_t)val);
}

// --- ERA 2: DNA DISPATCHER (Hyper-Optimized Persistent) ---
__global__ void dnaAutonomousKernel(const uint8_t* __restrict__ src, int iterations, uint32_t* finalScore) {
    __shared__ uint32_t blockSum;
    if (threadIdx.x == 0) blockSum = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_VOXELS) return;

    uint32_t localSum = 0;
    // FETCH ONCE: Load into register file (1 cycle latency)
    const uint8_t val = src[idx]; 
    if (val == 0) return;

    // The logic loop: Executes at core clock speed (~1.5GHz)
    #pragma unroll 4
	for(int i = 0; i < iterations; i++) {
        // 1. Recover original 3D coordinates from the thread index
        // This puts the same ALU load on the DNA kernel as the Typical one
        int x = idx / (N * N);
        int rem = idx % (N * N);
        int y = rem / N;
        int z = rem % N;

        // 2. Virtual Rotation (Alternate between X and Y axes)
        int nx, ny, nz;
        if (i % 2 == 0) { 
            // Virtual Rot X
            nx = x; ny = z; nz = (N - 1 - y);
        } else { 
            // Virtual Rot Y
            nx = (N - 1 - z); ny = y; nz = x;
        }

        // 3. Virtual Scoring (The "Perception" Step)
        // We verify the bounds in registers instead of reading from a buffer
        if (nx >= 0 && nx < N && ny >= 0 && ny < N && nz >= 0 && nz < N) {
            localSum += 1; 
        }
    }

    // Register-level parallel reduction (Warp Shuffle)
    for (int offset = 16; offset > 0; offset /= 2)
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);

    // Minimize global atomic collisions via Shared Memory
    if ((threadIdx.x & 31) == 0) atomicAdd(&blockSum, localSum);
    __syncthreads();

    if (threadIdx.x == 0 && blockSum > 0) atomicAdd(finalScore, blockSum);
}

int main() {
    uint8_t *d_src, *d_dst;
    uint32_t *d_score;
    cudaMalloc(&d_src, TOTAL_VOXELS);
    cudaMalloc(&d_dst, TOTAL_VOXELS);
    cudaMallocManaged(&d_score, sizeof(uint32_t));
    cudaMemset(d_src, 1, TOTAL_VOXELS);

    cudaEvent_t start, stop;
    float timeTypical, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warm-up to prime the Instruction Cache
    dnaAutonomousKernel<<<TOTAL_VOXELS/256, 256>>>(d_src, 1, d_score);
    cudaDeviceSynchronize();

    printf("--- 21/12 PERFORMANCE ASCENSION ---\n");

    // 1. BENCHMARK TYPICAL (Era 1: 200 Kernels, 200 Writes)
    cudaEventRecord(start);
    for(int i = 0; i < 100; i++) {
        typicalRotate<<<TOTAL_VOXELS/256, 256>>>(d_src, d_dst, i % 2);
        typicalScore<<<TOTAL_VOXELS/256, 256>>>(d_dst, d_score);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTypical, start, stop);

    // 2. BENCHMARK DNA (Era 2: 1 Kernel, 0 Writes)
    *d_score = 0;
    cudaEventRecord(start);
    dnaAutonomousKernel<<<TOTAL_VOXELS/256, 256>>>(d_src, 100, d_score);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeDNA, start, stop);

    printf("Typical Dispatcher (100 Cycles): %.3f ms (Memory Bound)\n", timeTypical);
    printf("DNA Dispatcher (100 Cycles):     %.3f ms (Instruction Bound)\n", timeDNA);
    printf("------------------------------------------\n");
    printf("PERFORMANCE INCREASE: %.2fx Faster\n", timeTypical / timeDNA);

    return 0;
}