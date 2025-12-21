%%writefile cuboid_ops.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cuboid_stride3_kernel(const int8_t* volume, int* matches, int N) {
    int cubes_per_dim = N / 3;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (cubes_per_dim * cubes_per_dim * cubes_per_dim)) return;

    // Calculate the "Structural Center" for this thread
    int cz = (tid / (cubes_per_dim * cubes_per_dim)) * 3 + 1;
    int cy = ((tid / cubes_per_dim) % cubes_per_dim) * 3 + 1;
    int cx = (tid % cubes_per_dim) * 3 + 1;

    // Structural T-Shape check: Only touch 5 specific memory addresses
    if (volume[cz*N*N + cy*N + cx] &&         // Center
        volume[(cz-1)*N*N + cy*N + cx] &&     // Top
        volume[(cz+1)*N*N + cy*N + cx] &&     // Bottom
        volume[cz*N*N + cy*N + (cx-1)] &&     // Left
        volume[cz*N*N + cy*N + (cx+1)])       // Right
    {
        atomicAdd(matches, 1);
    }
}

torch::Tensor cuboid_search(torch::Tensor volume) {
    int N = volume.size(0);
    auto matches = torch.zeros({1}, torch::dtype(torch.kInt32).device(volume.device()));
    int cubes_per_dim = N / 3;
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    
    cuboid_stride3_kernel<<<(total_cubes + 255) / 256, 256>>>(
        volume.data_ptr<int8_t>(), matches.data_ptr<int>(), N);
    
    return matches;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuboid_search", &cuboid_search, "Structural T-shape search");
}