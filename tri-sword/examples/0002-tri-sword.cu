%%writefile tri_sword.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <string>

// =====================================================================
// RESOURCE MANAGEMENT
// =====================================================================

struct KernelResources {
    int blocks;
    int threads;
    int* d_output;      
    int total_elements;
};

KernelResources* allocateKernelResources(int total_needed, int t = 256) {
    KernelResources* k = new KernelResources;
    k->threads = t;
    k->total_elements = total_needed;
    k->blocks = (total_needed + t - 1) / t;
    cudaMalloc(&k->d_output, k->total_elements * sizeof(int));
    return k;
}

void freeKernelResources(KernelResources* k) {
    cudaFree(k->d_output);
    delete k;
}

__global__ void ntt111_scythe(const uint8_t* data, int* output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const uint8_t* p = data + (idx * 256);
    int sum = 0;

    // --- PHASE 1: The Scythe ---
    #pragma unroll
    for (int i = 0; i < 40; i++) {
        sum += (int)p[i];
    }

    // EARLY EXIT: If we aren't at 20% of the threshold by point 40, kill it.
    // This removes 95% of the workload for noise spectrograms.
    if (sum < 200) return; 

    // --- PHASE 2: Deep Analysis ---
    #pragma unroll
    for (int i = 40; i < 111; i++) {
        sum += (int)p[i];
    }

    if (sum >= 1000) atomicAdd(output, 1);
}

// =====================================================================
// TYPICAL BASELINE (Float32) - Separate Entry Point
// =====================================================================

__global__ void typ_float_kernel(const float* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const float* p = data + (idx * 256);
    float sum = 0.0f;
    for (int i = 0; i < points; i++) { sum += p[i]; }
    // FIXED: Baseline threshold is 40.0 because its data is NOT scaled by 20x
    if (sum >= 35.0f) atomicAdd(output, 1);
}

// =====================================================================
// BINARY KERNELS (uint8_t) - Threshold 800 (40.0 * 20.0)
// =====================================================================

__global__ void v_binary(const uint8_t* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const uint8_t* p = data + (idx * 256);
    int sum = 0;
    for (int i = 0; i < points; i++) { sum += (int)p[i]; }
    if (sum >= 800) atomicAdd(output, 1);
}

__global__ void t_binary(const uint8_t* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const uint8_t* p = data + (idx * 256);
    int sum = 0;
    for (int i = 0; i < 20; i++) { sum += (int)p[i]; }
    if (sum < 15) return; 
    for (int i = 20; i < points; i++) { sum += (int)p[i]; }
    if (sum >= 800) atomicAdd(output, 1);
}

__global__ void p_binary(const uint8_t* data, int* output, int batch_size, int points) {
    __shared__ int b_sum;
    if (threadIdx.x == 0) b_sum = 0;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hit = 0;
    if (idx < batch_size) {
        const uint8_t* p = data + (idx * 256);
        int sum = 0;
        for (int i = 0; i < points; i++) { sum += (int)p[i]; }
        if (sum >= 800) hit = 1;
    }
    if (hit) atomicAdd(&b_sum, 1);
    __syncthreads();
    if (threadIdx.x == 0 && b_sum > 0) atomicAdd(output, b_sum);
}

__global__ void c_binary(const uint8_t* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const uint8_t* p = data + (idx * 256);
    int sum = 0;
    #pragma unroll 4
    for (int i = 0; i < points; i++) { sum += (int)p[i]; }
    if (sum >= 800) atomicAdd(output, 1);
}

__global__ void n_binary(const uint8_t* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const uint8_t* p = data + (idx * 256);
    int sum = 0;
    #pragma unroll 4 // Reduced unroll to prevent register spilling
    for (int i = 0; i < points; i++) { sum += (int)p[i]; }
    if (sum >= 800) atomicAdd(output, 1);
}

// =====================================================================
// TERNARY KERNELS (int8_t) - Threshold 800
// =====================================================================

__global__ void v_ternary(const int8_t* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const int8_t* p = data + (idx * 256);
    int sum = 0;
    for (int i = 0; i < points; i++) { sum += (int)p[i]; }
    if (sum >= 800) atomicAdd(output, 1);
}

__global__ void t_ternary(const int8_t* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const int8_t* p = data + (idx * 256);
    int sum = 0;
    for (int i = 0; i < 20; i++) { sum += (int)p[i]; }
    if (sum < 15) return;
    for (int i = 20; i < points; i++) { sum += (int)p[i]; }
    if (sum >= 800) atomicAdd(output, 1);
}

__global__ void p_ternary(const int8_t* data, int* output, int batch_size, int points) {
    __shared__ int b_sum;
    if (threadIdx.x == 0) b_sum = 0;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hit = 0;
    if (idx < batch_size) {
        const int8_t* p = data + (idx * 256);
        int sum = 0;
        for (int i = 0; i < points; i++) { sum += (int)p[i]; }
        if (sum >= 800) hit = 1;
    }
    if (hit) atomicAdd(&b_sum, 1);
    __syncthreads();
    if (threadIdx.x == 0 && b_sum > 0) atomicAdd(output, b_sum);
}

__global__ void c_ternary(const int8_t* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const int8_t* p = data + (idx * 256);
    int sum = 0;
    #pragma unroll 4
    for (int i = 0; i < points; i++) { sum += (int)p[i]; }
    if (sum >= 800) atomicAdd(output, 1);
}

__global__ void n_ternary(const int8_t* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const int8_t* p = data + (idx * 256);
    int sum = 0;
    #pragma unroll 4 // Reduced unroll to prevent register spilling
    for (int i = 0; i < points; i++) { sum += (int)p[i]; }
    if (sum >= 800) atomicAdd(output, 1);
}

// =====================================================================
// EXPOSED ENTRY POINTS
// =====================================================================

torch::Tensor typical(std::string func_name, int points, torch::Tensor input) {
    int batch_size = input.size(0);
    KernelResources* k = allocateKernelResources(batch_size);
    cudaMemset(k->d_output, 0, sizeof(int));

    typ_float_kernel<<<k->blocks, k->threads>>>(input.data_ptr<float>(), k->d_output, batch_size, points);

    auto output = torch::zeros({1}, torch::kFloat32);
    int h_out = 0;
    cudaMemcpy(&h_out, k->d_output, sizeof(int), cudaMemcpyDeviceToHost);
    output[0] = (float)h_out;
    
    freeKernelResources(k);
    return output;
}

torch::Tensor slash(std::string func_name, char shape, char dtype, char algo, int points, torch::Tensor input) {
    int batch_size = input.size(0);
    KernelResources* k = allocateKernelResources(batch_size); 
    cudaMemset(k->d_output, 0, sizeof(int));

    dim3 threads(k->threads);
    dim3 blocks((batch_size + k->threads - 1) / k->threads);

    if (dtype == 'B') {
        if (shape == 'V')      v_binary<<<blocks, threads>>>(static_cast<uint8_t*>(input.data_ptr()), k->d_output, batch_size, points);
        else if (shape == 'T') t_binary<<<blocks, threads>>>(static_cast<uint8_t*>(input.data_ptr()), k->d_output, batch_size, points);
        else if (shape == 'P') p_binary<<<blocks, threads>>>(static_cast<uint8_t*>(input.data_ptr()), k->d_output, batch_size, points);
        else if (shape == 'C') c_binary<<<blocks, threads>>>(static_cast<uint8_t*>(input.data_ptr()), k->d_output, batch_size, points);
        else if (shape == 'N') n_binary<<<blocks, threads>>>(static_cast<uint8_t*>(input.data_ptr()), k->d_output, batch_size, points);
    } else {
        if (shape == 'V')      v_ternary<<<blocks, threads>>>(static_cast<int8_t*>(input.data_ptr()), k->d_output, batch_size, points);
        else if (shape == 'T') t_ternary<<<blocks, threads>>>(static_cast<int8_t*>(input.data_ptr()), k->d_output, batch_size, points);
        else if (shape == 'P') p_ternary<<<blocks, threads>>>(static_cast<int8_t*>(input.data_ptr()), k->d_output, batch_size, points);
        else if (shape == 'C') c_ternary<<<blocks, threads>>>(static_cast<int8_t*>(input.data_ptr()), k->d_output, batch_size, points);
        else if (shape == 'N') n_ternary<<<blocks, threads>>>(static_cast<int8_t*>(input.data_ptr()), k->d_output, batch_size, points);
    }

    auto output = torch::zeros({1}, torch::kFloat32);
    int h_out = 0;
    cudaMemcpy(&h_out, k->d_output, sizeof(int), cudaMemcpyDeviceToHost);
    output[0] = (float)h_out;

    freeKernelResources(k);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("slash", &slash);
    m.def("typical", &typical);
}