%%writefile tri_sword.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

struct SwordHandle {
    int max_batch_size;
    int threads;
};

intptr_t sharpen(int max_batch_size, int t = 256) {
    SwordHandle* s = new SwordHandle;
    s->threads = t;
    s->max_batch_size = max_batch_size;
    return reinterpret_cast<intptr_t>(s);
}

void sheath(intptr_t handle) {
    SwordHandle* s = reinterpret_cast<SwordHandle*>(handle);
    if (s) delete s;
}

// RULE: Fast Local Solver using L1 (Shared Memory)
__global__ void d_ternary_maze_local(int8_t* grid, int batch_size, int dim) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (unsigned int)batch_size) return;

    int grid_size = dim * dim;
    int8_t* cell_ptr = grid + (size_t)b * grid_size;
    
    extern __shared__ int8_t shared_data[];
    // FIX: Dynamic offset based on actual grid size to prevent Illegal Access
    int8_t* local_state = &shared_data[threadIdx.x * grid_size]; 
    
    for(int i = 0; i < grid_size; i++) { local_state[i] = cell_ptr[i]; }

    bool changed = true;
    for (int g = 0; g < dim * 2 && changed; g++) {
        changed = false;
        for (int i = 0; i < grid_size; i++) {
            int8_t current = local_state[i];
            if (current == -1) { local_state[i] = 1; changed = true; continue; }
            if (current != 0) continue;

            int x = i % dim, y = i / dim;
            bool hit = (x > 0 && local_state[i - 1] == -1) ||
                       (x < dim - 1 && local_state[i + 1] == -1) ||
                       (y > 0 && local_state[i - dim] == -1) ||
                       (y < dim - 1 && local_state[i + dim] == -1);
            
            if (hit) { local_state[i] = -1; changed = true; }
        }
    }
    for(int i = 0; i < grid_size; i++) { cell_ptr[i] = local_state[i]; }
}

// RULE: Branchless Large-Grid Solver (Global VRAM)
__global__ void d_ternary_maze_inplace(int8_t* grid, int batch_size, int dim) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (unsigned int)batch_size) return;

    int grid_size = dim * dim;
    int8_t* cell_ptr = grid + (size_t)b * grid_size;
    bool changed = true;

    for (int g = 0; g < dim * 2 && changed; g++) {
        changed = false;
        int start = g & 1; // RULE: Bitwise parity instead of if/else
        for (int i = start; i < grid_size; i += 2) {
            int8_t current = cell_ptr[i];
            if (current == -1) { cell_ptr[i] = 1; changed = true; continue; }
            if (current != 0) continue;

            int x = i % dim, y = i / dim;
            if ((x > 0 && cell_ptr[i-1] == -1) || (x < dim-1 && cell_ptr[i+1] == -1) ||
                (y > 0 && cell_ptr[i-dim] == -1) || (y < dim-1 && cell_ptr[i+dim] == -1)) {
                cell_ptr[i] = -1; changed = true;
            }
        }
    }
}

torch::Tensor slash(intptr_t handle, char shape, char dtype, int dim, torch::Tensor input) {
    SwordHandle* s = reinterpret_cast<SwordHandle*>(handle);
    int batch_size = input.size(0);
    int8_t* d_ptr = static_cast<int8_t*>(input.data_ptr());
    
    dim3 threads(s->threads);
    dim3 blocks((batch_size + s->threads - 1) / s->threads);

    int grid_size = dim * dim;
    size_t shared_needed = s->threads * grid_size;

    // SILICON GUARDRAIL: Tesla T4 has 48KB Shared Memory limit
    if (shared_needed <= 48000) {
        d_ternary_maze_local<<<blocks, threads, shared_needed>>>(d_ptr, batch_size, dim);
    } else {
        d_ternary_maze_inplace<<<blocks, threads>>>(d_ptr, batch_size, dim);
    }

    return input; // RULE: Zero memory movement back to host
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sharpen", &sharpen);
    m.def("sheath", &sheath);
    m.def("slash", &slash);
}