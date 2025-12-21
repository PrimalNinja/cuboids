%%writefile cuboid_ops.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct Cube81 {
    int8_t points[27];
    __device__ bool hasTShape() const {
        return (points[4] && points[13] && points[22]) && (points[12] && points[13] && points[14]);
    }
};

__global__ void cuboid_kernel(const int8_t* volume, int* matches, int N, int cubes_per_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    if (tid >= total_cubes) return;
    int cx = (tid % cubes_per_dim) * 3;
    int cy = ((tid / cubes_per_dim) % cubes_per_dim) * 3;
    int cz = (tid / (cubes_per_dim * cubes_per_dim)) * 3;
    Cube81 cube;
    int idx = 0;
    for (int dz = 0; dz < 3; ++dz) {
        for (int dy = 0; dy < 3; ++dy) {
            for (int dx = 0; dx < 3; ++dx) {
                int vol_idx = (cz + dz) * N * N + (cy + dy) * N + (cx + dx);
                cube.points[idx++] = volume[vol_idx];
            }
        }
    }
    if (cube.hasTShape()) atomicAdd(matches, 1);
}

torch::Tensor cuboid_search(torch::Tensor volume) {
    int N = volume.size(0);
    int cubes_per_dim = N / 3;
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    auto matches = torch::zeros({1}, torch::dtype(torch::kInt32).device(volume.device()));
    cuboid_kernel<<<(total_cubes + 255) / 256, 256>>>(volume.data_ptr<int8_t>(), matches.data_ptr<int>(), N, cubes_per_dim);
    return matches;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuboid_search", &cuboid_search, "Search");
}