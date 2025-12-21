%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <iomanip>

// 1. TRADITIONAL VOXEL KERNEL: Processes entire N^3 volume
__global__ void traditionalVoxelKernel(const int8_t* data, int N, int* out_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int idx = x + (y * N) + (z * N * N);
        if (data[idx] > 0) atomicAdd(out_count, 1);
    }
}

// 2. FACE-BASED KERNEL: Processes only the 6 exterior faces (6 * N^2)

__global__ void faceKernel(const int8_t* data, int N, int* out_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = N * N;
    if (tid >= face_size * 6) return;

    int face_id = tid / face_size;
    int face_idx = tid % face_size;
    int i = face_idx / N;
    int j = face_idx % N;

    int vol_idx = 0;
    switch(face_id) {
        case 0: vol_idx = 0 * N * N + i * N + j; break;         // Front
        case 1: vol_idx = (N-1) * N * N + i * N + j; break;     // Back
        case 2: vol_idx = i * N * N + 0 * N + j; break;         // Bottom
        case 3: vol_idx = i * N * N + (N-1) * N + j; break;     // Top
        case 4: vol_idx = i * N * N + j * N + 0; break;         // Left
        case 5: vol_idx = i * N * N + j * N + (N-1); break;     // Right
    }

    if (data[vol_idx] > 0) atomicAdd(out_count, 1);
}

void run_test(int N) {
    size_t size = (size_t)N * N * N;
    int8_t *d_data;
    int *d_count, h_count = 0;
    
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_data, 1, size);
    cudaMemset(d_count, 0, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmarking Traditional (N^3)
    dim3 block(8, 8, 8);
    dim3 grid((N+7)/8, (N+7)/8, (N+7)/8);
    cudaEventRecord(start);
    traditionalVoxelKernel<<<grid, block>>>(d_data, N, d_count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float voxel_ms;
    cudaEventElapsedTime(&voxel_ms, start, stop);

    // Reset count for second test
    cudaMemset(d_count, 0, sizeof(int));

    // Benchmarking Face (6 * N^2)
    int total_face_threads = 6 * N * N;
    cudaEventRecord(start);
    faceKernel<<<(total_face_threads + 255)/256, 256>>>(d_data, N, d_count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float face_ms;
    cudaEventElapsedTime(&face_ms, start, stop);

    // Retrieve result before freeing GPU memory
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Consolidated Output
    std::cout << "N=" << std::setw(3) << N 
              << " | Count: " << std::setw(8) << h_count
              << " | Trad: " << std::fixed << std::setprecision(4) << std::setw(8) << voxel_ms << "ms"
              << " | Face: " << std::setw(8) << face_ms << "ms" 
              << " | Speedup: " << (voxel_ms / face_ms) << "x" << std::endl;

    cudaFree(d_data); 
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "Scaling Analysis: Voxel (O(N^3)) vs Faces (O(N^2))\n";
    std::cout << "--------------------------------------------------\n";
    run_test(64);
    run_test(128);
    run_test(256);
    run_test(512);
    return 0;
}