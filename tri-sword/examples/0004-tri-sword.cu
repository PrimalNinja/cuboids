%%writefile tri_sword.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

struct SwordHandle {
    uint8_t* d_grid_b; 
    int max_batch_size;
    int threads;
};

intptr_t sharpen(int max_batch_size, int t = 256) {
    SwordHandle* s = new SwordHandle;
    s->threads = t;
    s->max_batch_size = max_batch_size;
    cudaMalloc(&s->d_grid_b, max_batch_size * 256); 
    return reinterpret_cast<intptr_t>(s);
}

void sheath(intptr_t handle) {
    SwordHandle* s = reinterpret_cast<SwordHandle*>(handle);
    if (s) {
        cudaFree(s->d_grid_b);
        delete s;
    }
}

// --- DNA TURING KERNEL: PERSISTENT SILICON LOOP ---
__global__ void d_dna_turing(uint8_t* grid, int batch_size, int generations) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    uint8_t* cell_ptr = grid + (b * 256);
    uint8_t local_state[256];

    // FETCH: Single VRAM Read
    #pragma unroll
    for(int i = 0; i < 256; i++) local_state[i] = cell_ptr[i];

    // PERSISTENCE: Evolution stays in Registers/L1
    for (int g = 0; g < generations; g++) {
        uint8_t next_state[256];
        #pragma unroll
        for (int i = 0; i < 256; i++) {
            int x = i & 15;
            int y = i >> 4;
            int neighbors = 0;

            #pragma unroll
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = (x + dx + 16) & 15;
                    int ny = (y + dy + 16) & 15;
                    neighbors += local_state[(ny << 4) | nx];
                }
            }
            uint8_t current = local_state[i];
            next_state[i] = (neighbors == 3) || (current && neighbors == 2);
        }
        #pragma unroll
        for(int i = 0; i < 256; i++) local_state[i] = next_state[i];
    }

    // COMMIT: Single VRAM Write
    #pragma unroll
    for(int i = 0; i < 256; i++) cell_ptr[i] = local_state[i];
}

torch::Tensor slash(intptr_t handle, char shape, char dtype, int points, torch::Tensor input) {
    SwordHandle* s = reinterpret_cast<SwordHandle*>(handle);
    int batch_size = input.size(0);
    dim3 threads(s->threads);
    dim3 blocks((batch_size + s->threads - 1) / s->threads);

    if (shape == 'D') {
        uint8_t* d_ptr = static_cast<uint8_t*>(input.data_ptr());
        d_dna_turing<<<blocks, threads>>>(d_ptr, batch_size, points);
        // FIXED: added () to .device()
        return torch::zeros({1}, torch::kInt8).to(input.device());
    }
    // FIXED: added () to .device()
    return torch::zeros({1}, torch::kInt8).to(input.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sharpen", &sharpen);
    m.def("sheath", &sheath);
    m.def("slash", &slash);
}