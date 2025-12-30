%%writefile tri_sword.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>
#include <string>

// =====================================================================
// RESOURCE MANAGEMENT (The Persistent Handle)
// =====================================================================

struct SwordHandle {
    int* d_output;      
    int max_batch_size;
    int threads;
};

intptr_t sharpen(int max_batch_size, int t = 256) {
    SwordHandle* s = new SwordHandle;
    s->threads = t;
    s->max_batch_size = max_batch_size;
    cudaMalloc(&s->d_output, sizeof(int)); 
    return reinterpret_cast<intptr_t>(s);
}

void sheath(intptr_t handle) {
    SwordHandle* s = reinterpret_cast<SwordHandle*>(handle);
    if (s) {
        cudaFree(s->d_output);
        delete s;
    }
}

// =====================================================================
// BINARY KERNELS (uint8_t) - Threshold 800
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
    #pragma unroll 4
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
    #pragma unroll 4
    for (int i = 0; i < points; i++) { sum += (int)p[i]; }
    if (sum >= 800) atomicAdd(output, 1);
}

// =====================================================================
// SPECIALTY KERNELS (Scythe & Typical)
// =====================================================================

__global__ void ntt111_scythe(const uint8_t* data, int* output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const uint8_t* p = data + (idx * 256);
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 40; i++) { sum += (int)p[i]; }
    if (sum < 200) return; 
    #pragma unroll
    for (int i = 40; i < 111; i++) { sum += (int)p[i]; }
    if (sum >= 1000) atomicAdd(output, 1);
}

__global__ void typ_float_kernel(const float* data, int* output, int batch_size, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const float* p = data + (idx * 256);
    float sum = 0.0f;
    for (int i = 0; i < points; i++) { sum += p[i]; }
    if (sum >= 35.0f) atomicAdd(output, 1);
}

// =====================================================================
// DISPATCHER (The Slash)
// =====================================================================

torch::Tensor slash(intptr_t handle, char shape, char dtype, int points, torch::Tensor input) {
    SwordHandle* s = reinterpret_cast<SwordHandle*>(handle);
    int batch_size = input.size(0);
    
    cudaMemset(s->d_output, 0, sizeof(int));
    dim3 threads(s->threads);
    dim3 blocks((batch_size + s->threads - 1) / s->threads);

    if (dtype == 'B') { // Binary (uint8)
        uint8_t* d_ptr = static_cast<uint8_t*>(input.data_ptr());
        if (shape == 'V')      v_binary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
        else if (shape == 'T') t_binary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
        else if (shape == 'P') p_binary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
        else if (shape == 'C') c_binary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
        else if (shape == 'N') n_binary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
        else if (shape == 'S') ntt111_scythe<<<blocks, threads>>>(d_ptr, s->d_output, batch_size);
    } 
    else if (dtype == 'T') { // Ternary (int8)
        int8_t* d_ptr = static_cast<int8_t*>(input.data_ptr());
        if (shape == 'V')      v_ternary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
        else if (shape == 'T') t_ternary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
        else if (shape == 'P') p_ternary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
        else if (shape == 'C') c_ternary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
        else if (shape == 'N') n_ternary<<<blocks, threads>>>(d_ptr, s->d_output, batch_size, points);
    }
    else if (dtype == 'F') { // Typical (float32)
        typ_float_kernel<<<blocks, threads>>>(input.data_ptr<float>(), s->d_output, batch_size, points);
    }

    int h_out = 0;
    cudaMemcpy(&h_out, s->d_output, sizeof(int), cudaMemcpyDeviceToHost);
    return torch::tensor({(float)h_out}, torch::kFloat32);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sharpen", &sharpen);
    m.def("sheath", &sheath);
    m.def("slash", &slash);
}