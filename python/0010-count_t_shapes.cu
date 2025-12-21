%%writefile cuboid_ops.cu
#include <torch/extension.h>

__global__ void cube81_tshape_kernel(const int8_t* volume, int* matches, int N, int total_cubes) {
    extern __shared__ int shared[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int lane = threadIdx.x;
    
    int local_count = 0;
    
    int cubes_per_dim = N / 3;
    
    for (int cube_id = tid; cube_id < total_cubes; cube_id += stride) {
        int temp = cube_id;
        int cz = (temp / (cubes_per_dim * cubes_per_dim)) * 3;
        temp %= (cubes_per_dim * cubes_per_dim);
        int cy = (temp / cubes_per_dim) * 3;
        int cx = (temp % cubes_per_dim) * 3;
        
        bool p[27] = {false};
        int idx = 0;
        for (int dz = 0; dz < 3; ++dz) {
            for (int dy = 0; dy < 3; ++dy) {
                for (int dx = 0; dx < 3; ++dx) {
                    int vol_idx = (cz + dz) * N * N + (cy + dy) * N + (cx + dx);
                    p[idx++] = (volume[vol_idx] != 0);
                }
            }
        }
        
        if (p[13] && p[4] && p[22] && p[12] && p[14]) {
            local_count++;
        }
    }
    
    shared[lane] = local_count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane < s) {
            shared[lane] += shared[lane + s];
        }
        __syncthreads();
    }
    
    if (lane == 0) {
        atomicAdd(matches, shared[0]);
    }
}

torch::Tensor count_t_shapes(torch::Tensor volume) {
    TORCH_CHECK(volume.is_contiguous(), "Volume must be contiguous");
    TORCH_CHECK(volume.dim() == 3 && volume.scalar_type() == torch::kInt8, "Expected contiguous int8 (N,N,N)");
    
    int N = volume.size(0);
    TORCH_CHECK(N % 3 == 0, "N must be divisible by 3");
    
    int cubes_per_dim = N / 3;
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    
    auto matches = torch::zeros({1}, torch::kInt32).to(volume.device());  // Use int32
    
    int threads = 256;
    int blocks = (total_cubes + threads - 1) / threads;
    blocks = min(blocks, 65536);
    
    size_t shared_bytes = threads * sizeof(int);
    
    cube81_tshape_kernel<<<blocks, threads, shared_bytes>>>(
        volume.data_ptr<int8_t>(),
        matches.data_ptr<int>(),
        N,
        total_cubes
    );
    
    return matches;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_t_shapes", &count_t_shapes, "Count T-shapes in non-overlapping 3x3x3 cuboids");
}