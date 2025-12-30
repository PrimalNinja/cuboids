#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

struct KernelResources {
    int blocks;
    int threads;
    int* d_output;      // Using int for trit/ternary values and counts
    int total_elements;
};

KernelResources* allocateKernelResources(const std::vector<int>& element_groups, int t = 256) {
    KernelResources* k = new KernelResources;
    k->threads = t;
    int sum = 0;
    for (int count : element_groups) sum += count;
    k->total_elements = sum;
    k->blocks = (k->total_elements + k->threads - 1) / k->threads;
    cudaMalloc(&k->d_output, k->blocks * k->threads * sizeof(int));
    return k;
}

void freeKernelResources(KernelResources* k) {
    cudaFree(k->d_output);
    delete k;
}

// Cube27 structure for 3x3x3 voxel processing
struct Cube27 {
    int8_t points[27];
    
    __device__ bool hasShip() const {
        // A ship is detected if there's a line of 3+ occupied voxels
        // Check all possible 3-in-a-row patterns in the cube
        
        // Horizontal lines (z=constant, y=constant)
        for (int z = 0; z < 3; z++) {
            for (int y = 0; y < 3; y++) {
                int count = 0;
                for (int x = 0; x < 3; x++) {
                    int idx = z*9 + y*3 + x;
                    if (points[idx] > 0) count++;
                }
                if (count >= 3) return true;
            }
        }
        
        // Vertical lines (z=constant, x=constant)
        for (int z = 0; z < 3; z++) {
            for (int x = 0; x < 3; x++) {
                int count = 0;
                for (int y = 0; y < 3; y++) {
                    int idx = z*9 + y*3 + x;
                    if (points[idx] > 0) count++;
                }
                if (count >= 3) return true;
            }
        }
        
        // Depth lines (y=constant, x=constant)
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                int count = 0;
                for (int z = 0; z < 3; z++) {
                    int idx = z*9 + y*3 + x;
                    if (points[idx] > 0) count++;
                }
                if (count >= 3) return true;
            }
        }
        
        return false;
    }
    
    __device__ int countCollisions() const {
        // Count voxels where value > 1 (multiple ships)
        int collisions = 0;
        for (int i = 0; i < 27; i++) {
            if (points[i] > 1) collisions++;
        }
        return collisions;
    }
};

// Callback: Count ships using Cube27 structural approach
__global__ void countDiscoveryOnes(float* input, int* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cubes_per_dim = N / 3;
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    
    if (tid >= total_cubes) return;
    
    // Which cube are we processing?
    int cube_x = (tid % cubes_per_dim) * 3;
    int cube_y = ((tid / cubes_per_dim) % cubes_per_dim) * 3;
    int cube_z = (tid / (cubes_per_dim * cubes_per_dim)) * 3;
    
    // Extract Cube27 from input volume
    Cube27 cube;
    int idx = 0;
    for (int dz = 0; dz < 3; dz++) {
        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                int vol_idx = (cube_z+dz)*N*N + (cube_y+dy)*N + (cube_x+dx);
                float val = input[vol_idx];
                cube.points[idx++] = (int8_t)(val > 0.5f ? (val > 1.5f ? 2 : 1) : 0);
            }
        }
    }
    
    // Detect ships in this cube
    if (cube.hasShip()) {
        atomicAdd(&output[0], 1);  // Ship count
    }
    
    // Count collisions in this cube
    int collisions = cube.countCollisions();
    if (collisions > 0) {
        atomicAdd(&output[1], collisions);  // Collision count
    }
}

// Callback: Test collision detection (separate from ship counting)
__global__ void testDiscoveryOneCollisions(float* input, int* output, int N, int offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    // Simple collision test: count voxels with value > 1.5
    if (input[tid] > 1.5f) {
        atomicAdd(&output[offset], 1);
    }
}

// Orchestration callback for Discovery One detection
void doDiscoveryOneDetection(KernelResources* k, float* d_input, int base_n, int start_offset) {
    // Calculate N (cube root of base_n)
    int N = (int)round(cbrt((double)base_n));
    
    // Launch ship counting callback
    int cubes_per_dim = N / 3;
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    
    countDiscoveryOnes<<<(total_cubes + k->threads - 1) / k->threads, k->threads>>>(
        d_input, k->d_output + start_offset, N
    );
    
    cudaDeviceSynchronize();
}

torch::Tensor slash(std::string func_name, char shape_char, char datatype_char, 
                    char algorithm_char, int permutation, torch::Tensor input) {
    auto input_flat = input.flatten();
    int base_n = input_flat.size(0);
    
    // For Discovery One detection, we need space for:
    // [0] = ship count
    // [1] = collision count
    std::vector<int> element_groups;
    if (func_name == "discoveryone_detection") {
        element_groups.push_back(2);  // 2 output values
    } else {
        element_groups.push_back(base_n);
    }

    KernelResources* k = allocateKernelResources(element_groups);
    cudaMemset(k->d_output, 0, k->total_elements * sizeof(int));

    if (func_name == "discoveryone_detection") {
        doDiscoveryOneDetection(k, input_flat.data_ptr<float>(), base_n, 0);
    }

    // Copy results back to CPU as float tensor (for consistency with PyTorch)
    auto output = torch::zeros({k->total_elements}, torch::kFloat32);
    std::vector<int> h_output(k->total_elements);
    cudaMemcpy(h_output.data(), k->d_output, k->total_elements * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < k->total_elements; i++) {
        output[i] = (float)h_output[i];
    }
    
    freeKernelResources(k);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("slash", &slash);
}