%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 32
#define TOTAL (N * N * N)

// --- TRADITIONAL: Affine Transformation ---
// Mimics standard GPU sampling with floating point trig and matrix math
__global__ void traditionalRotate(float* in, float* out) {
    int x = threadIdx.x; int y = threadIdx.y; int z = blockIdx.z;
    
    // Traditional matrix-based rotation requires sin/cos or 4x4 multiplication
    float angle = 1.5708f; // 90 degrees
    float s = sinf(angle);
    float c = cosf(angle);

    // Rotation around center logic
    float cx = x - 16.0f;
    float cy = y - 16.0f;
    
    int nx = (int)(cx * c - cy * s + 16.0f);
    int ny = (int)(cx * s + cy * c + 16.0f);

    if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
        out[nx + (ny * N) + (z * N * N)] = in[x + (y * N) + (z * N * N)];
    }
}

// --- DNA PERSISTENT: Kinetic Bit-Shift ---
// Uses hard-coded integer swizzling (Zero Trig)
__global__ void dnaRotate(int8_t* in, int8_t* out) {
    int x = threadIdx.x; int y = threadIdx.y; int z = blockIdx.z;
    
    // New Y = Z, New Z = -Y (Centered Mapping)
    int ny = z;
    int nz = 31 - y;

    out[x + (ny * N) + (nz * N * N)] = in[x + (y * N) + (z * N * N)];
}

int main() {
    float *d_inF, *d_outF;
    int8_t *d_inT, *d_outT;

    cudaMallocManaged(&d_inF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_outF, TOTAL * sizeof(float));
    cudaMallocManaged(&d_inT, TOTAL);
    cudaMallocManaged(&d_outT, TOTAL);

    dim3 threads(N, N, 1);
    dim3 blocks(1, 1, N);

    std::cout << "--- 21/12 RELEASE: FILE 0048 (KINETIC RACE) ---" << std::endl;

    // 1. TRADITIONAL Benchmark (Trig + Floats)
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) traditionalRotate<<<blocks, threads>>>(d_inF, d_outF);
    cudaDeviceSynchronize();
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT Benchmark (Integer Swizzle)
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++) dnaRotate<<<blocks, threads>>>(d_inT, d_outT);
    cudaDeviceSynchronize();
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (Matrix/Trig): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Swizzle):  " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "SPATIAL EFFICIENCY GAP: " << (trad_ms / dna_ms) << "x" << std::endl;

    cudaFree(d_inF); cudaFree(d_outF);
    cudaFree(d_inT); cudaFree(d_outT);
    return 0;
}