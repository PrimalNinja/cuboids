%%writefile engine.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void structural_query_kernel(const uint8_t* data, int* results, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // PROOF OF LIFE: We aren't just summing; we are checking 
        // a specific structural property (e.g., neighbor is active)
        if (data[idx] == 1 && data[idx + 1] == 1) {
            atomicAdd(results, 1); 
        }
    }
}

// The "API Entry Point"
int run_structural_query(torch::Tensor input) {
    auto results = torch::zeros({1}, torch::kInt32).to(input.device());
    int N = input.numel();
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    structural_query_kernel<<<blocks, threads>>>(
        input.data_ptr<uint8_t>(), 
        results.data_ptr<int>(), 
        N
    );
    
    return results.item<int>();
}

// BINDING: This creates the "Custom API"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("query", &run_structural_query, "Expert Structural Query");
}