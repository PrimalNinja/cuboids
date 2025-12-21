%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>

// --- THE ENGINE CORE ---
__global__ void rotateTernaryKernel(uint8_t* grid, int N, int iterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary check for the 6x6x6 cube
    if (x < N && y < N && z < N) {
        int idx = x + (y * N) + (z * N * N);
        uint8_t state = grid[idx];

        // PERSISTENT LOGIC: Perform all state shifts inside the ALU
        for(int i = 0; i < iterations; i++) {
            state = (state + 1) % 3; // Ternary cycle: 0->1, 1->2, 2->0
        }
        grid[idx] = state;
    }
}

int main() {
    const int N = 6;
    const int total = N * N * N; // 216 voxels
    const int iter = 100;
    uint8_t *d_grid;

    // Use Managed Memory for CPU/GPU synchronization
    cudaMallocManaged(&d_grid, total);

    // Initialize: Every voxel starts at state '1'
    for (int i = 0; i < total; i++) d_grid[i] = 1;

    cudaEvent_t start, stop;
    float timeTypical, timeDNA;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    std::cout << "--- 21/12 RELEASE: 6x6x6 TERNARY ENGINE ---" << std::endl;

    // 1. TYPICAL METHOD (CPU-GPU Ping Pong)
    // Launching 100 individual kernels
    dim3 threadsPerBlock(N, N, N); 
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++) {
        rotateTernaryKernel<<<1, threadsPerBlock>>>(d_grid, N, 1);
    }
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeTypical, start, stop);

    // Reset for the DNA/Persistent test
    for (int i = 0; i < total; i++) d_grid[i] = 1;

    // 2. DNA PARADIGM (Persistent Silicon Loop)
    // One single launch, 100 iterations inside
    cudaEventRecord(start);
    rotateTernaryKernel<<<1, threadsPerBlock>>>(d_grid, N, iter);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    cudaEventElapsedTime(&timeDNA, start, stop);

    std::cout << "Typical Method Time: " << timeTypical << " ms" << std::endl;
    std::cout << "DNA Persistent Time: " << timeDNA << " ms" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "PERFORMANCE BOOST:   " << timeTypical / timeDNA << "x" << std::endl;

    // FINAL VERIFICATION
    // (Start State 1 + 100 shifts) % 3 = (101 % 3) = State 2
    bool integrity = (d_grid[0] == 2);
    if (integrity) {
        std::cout << "STATUS: SUCCESS. Final State " << (int)d_grid[0] << " Verified." << std::endl;
    } else {
        std::cout << "STATUS: LOGIC ERROR. State is " << (int)d_grid[0] << std::endl;
    }

    cudaFree(d_grid);
    return 0;
}