%%writefile ultimate_benchmark_v2.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define FREQ_BINS 128
#define TIME_SLICES 2
#define TOTAL_POINTS (FREQ_BINS * TIME_SLICES)

// ============================================================================
// APPROACH 8: CUBE81 (The secret weapon - 3^4 = 81 bytes)
// ============================================================================
struct __align__(128) Cube81 {
    int8_t points[81];  // 3×3×3×3 = perfect GPU alignment!
    int8_t padding[47]; // Pad to 128 bytes (cache line)
    
    __device__ bool hasKickPattern() const {
        // Process in chunks of 27 (3 layers of 3×3×3)
        int8_t bass_layer = 0;    // First 27 points
        int8_t mid_layer = 0;     // Middle 27 points  
        int8_t high_layer = 0;    // Last 27 points
        
        #pragma unroll
        for (int i = 0; i < 27; i++) {
            bass_layer += points[i];
            mid_layer += points[i + 27];
            high_layer += points[i + 54];
        }
        
        // Ternary decision on each layer
        return (bass_layer > 9) && (mid_layer < 3) && (high_layer < -6);
    }
};

__global__ void cube81_detect(
    const float* spectrogram,
    int* detections,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    Cube81 cube;
    
    // Fill 81 points: 3 time slices × 27 frequency bands
    int cube_idx = 0;
    for (int t = 0; t < 3; t++) {  // Time (extended to 3 using interpolation)
        for (int z = 0; z < 3; z++) {  // Freq band groups
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    int f = (z * 3 + y) * (FREQ_BINS / 9) + x * (FREQ_BINS / 27);
                    int time_idx = (t * TIME_SLICES) / 3;
                    
                    if (f < FREQ_BINS && time_idx < TIME_SLICES) {
                        float energy = spectrogram[idx * TOTAL_POINTS + time_idx * FREQ_BINS + f];
                        cube.points[cube_idx] = (energy > 2.0f) ? 1 : (energy > 0.5f) ? 0 : -1;
                    } else {
                        cube.points[cube_idx] = 0;
                    }
                    cube_idx++;
                }
            }
        }
    }
    
    if (cube.hasKickPattern()) {
        atomicAdd(detections, 1);
    }
}

// ============================================================================
// APPROACH 9: OPTIMIZED CUBE27 (Cache-perfect 32 bytes)
// ============================================================================
struct __align__(32) OptimizedCube27 {
    int8_t points[27];
    int8_t padding[5];
    
    __device__ bool hasKickPattern() const {
        int8_t bass = 0, highs = 0;
        
        // Bottom layer (bass)
        #pragma unroll
        for (int i = 0; i < 9; i++) bass += points[i];
        
        if (bass <= 3) return false;  // Early exit
        
        // Top layer (highs)
        #pragma unroll
        for (int i = 18; i < 27; i++) highs += points[i];
        
        return (highs < -3);
    }
};

__global__ void optimized_cube27_detect(
    const float* spectrogram,
    int* detections,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    OptimizedCube27 cube;
    
    int cube_idx = 0;
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                int f_start = y * (FREQ_BINS / 3);
                int f_end = f_start + (FREQ_BINS / 3);
                int t = z % TIME_SLICES;
                
                float energy = 0.0f;
                for (int f = f_start; f < f_end && f < FREQ_BINS; f++) {
                    energy += spectrogram[idx * TOTAL_POINTS + t * FREQ_BINS + f];
                }
                
                cube.points[cube_idx++] = (energy > 2.0f) ? 1 : (energy > 0.5f) ? 0 : -1;
            }
        }
    }
    
    if (cube.hasKickPattern()) {
        atomicAdd(detections, 1);
    }
}

// ============================================================================
// APPROACH 10: MEMORY COALESCING TEST (Warp-aligned processing)
// ============================================================================
__global__ void coalesced_cube27_detect(
    const float* spectrogram,
    int* detections,
    int N
) {
    // Each warp processes 32 spectrograms simultaneously
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int idx = warp_id * 32 + lane_id;
    
    if (idx >= N) return;
    
    // All 32 threads in warp access memory together (coalesced!)
    __shared__ OptimizedCube27 s_cubes[32];
    
    // Each thread fills its own cube
    int cube_idx = 0;
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                int f_start = y * (FREQ_BINS / 3);
                int f_end = f_start + (FREQ_BINS / 3);
                int t = z % TIME_SLICES;
                
                float energy = 0.0f;
                for (int f = f_start; f < f_end && f < FREQ_BINS; f++) {
                    energy += spectrogram[idx * TOTAL_POINTS + t * FREQ_BINS + f];
                }
                
                s_cubes[lane_id].points[cube_idx++] = (energy > 2.0f) ? 1 : (energy > 0.5f) ? 0 : -1;
            }
        }
    }
    
    __syncwarp();
    
    if (s_cubes[lane_id].hasKickPattern()) {
        atomicAdd(detections, 1);
    }
}

// ============================================================================
// BINDINGS
// ============================================================================
torch::Tensor approach8_cube81(torch::Tensor spectrogram) {
    int N = spectrogram.size(0);
    auto detections = torch::zeros({1}, torch::kInt32).to(spectrogram.device());
    
    cube81_detect<<<(N + 255) / 256, 256>>>(
        spectrogram.data_ptr<float>(),
        detections.data_ptr<int>(),
        N
    );
    
    return detections;
}

torch::Tensor approach9_optimized_cube27(torch::Tensor spectrogram) {
    int N = spectrogram.size(0);
    auto detections = torch::zeros({1}, torch::kInt32).to(spectrogram.device());
    
    optimized_cube27_detect<<<(N + 255) / 256, 256>>>(
        spectrogram.data_ptr<float>(),
        detections.data_ptr<int>(),
        N
    );
    
    return detections;
}

torch::Tensor approach10_coalesced(torch::Tensor spectrogram) {
    int N = spectrogram.size(0);
    auto detections = torch::zeros({1}, torch::kInt32).to(spectrogram.device());
    
    coalesced_cube27_detect<<<(N + 255) / 256, 256>>>(
        spectrogram.data_ptr<float>(),
        detections.data_ptr<int>(),
        N
    );
    
    return detections;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("approach8_cube81", &approach8_cube81, "Cube81 (81 bytes, 128-aligned)");
    m.def("approach9_optimized_cube27", &approach9_optimized_cube27, "Optimized Cube27 (32 bytes)");
    m.def("approach10_coalesced", &approach10_coalesced, "Coalesced Cube27 (warp-optimized)");
}