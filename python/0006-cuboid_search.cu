%%writefile cuboid_ops.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void cuboid_stride3_kernel(const int8_t* volume, int* matches, int N) {
    int cubes_per_dim = N / 3;
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_cubes) return;

    // Calculate grid coordinates (Top-Left-Front corner of the 3x3x3 block)
    int cz = (tid / (cubes_per_dim * cubes_per_dim)) * 3;
    int cy = ((tid / cubes_per_dim) % cubes_per_dim) * 3;
    int cx = (tid % cubes_per_dim) * 3;

    // Check 5 specific points for the T-Shape relative to the block
    // Vertical bar: (0,1,1), (1,1,1), (2,1,1)
    // Horizontal bar: (1,0,1), (1,1,1), (1,2,1)
    
    bool vertical = volume[(cz*N*N) + ((cy+1)*N) + (cx+1)] && 
                    volume[((cz+1)*N*N) + ((cy+1)*N) + (cx+1)] && 
                    volume[((cz+2)*N*N) + ((cy+1)*N) + (cx+1)];
                    
    bool horizontal = volume[((cz+1)*N*N) + (cy*N) + (cx+1)] && 
                      volume[((cz+1)*N*N) + ((cy+2)*N) + (cx+1)];

    if (vertical && horizontal) {
        atomicAdd(matches, 1);
    }
}

torch::Tensor cuboid_search(torch::Tensor volume) {
    const int N = volume.size(0);
    // Use a simpler way to create the tensor that NVCC likes
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(volume.device());
    torch::Tensor matches = torch::zeros({1}, options);
    
    int cubes_per_dim = N / 3;
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    int threads = 256;
    int blocks = (total_cubes + threads - 1) / threads;

    cuboid_stride3_kernel<<<blocks, threads>>>(
        volume.data_ptr<int8_t>(), 
        matches.data_ptr<int>(), 
        N
    );
    
    return matches;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuboid_search", &cuboid_search, "Expert Stride-3 Search");
}