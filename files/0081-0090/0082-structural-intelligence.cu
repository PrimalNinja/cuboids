=== COMPLEXITY VS QUANTITY EXPERIMENT ===
N=512, Total Voxels=134217728
Voxel Grid: 134217728 simple elements
Cube81 Grid: 4913000 complex elements (81 points each)

=== DEBUG: Placing test T-shape ===
Cube: (5,5,5)
Global center: (16,16,16)

Placing SMALL T-shape (fits in one 3×3×3 cube):
  Vertical at global: (31,31,30)
  Vertical at global: (31,31,31)
  Vertical at global: (31,31,32)
  Horizontal at global: (30,31,31)
  Horizontal at global: (31,31,31)
  Horizontal at global: (32,31,31)

Adding 5% random noise (skipping test cube region)...


1. VOXEL GRID APPROACH (Many simple elements):
   Time: 205.46 ms
   Elements processed: 134217728 voxels
   Operations per element: ~343 pattern checks
   Total operations: -1207959552
   Matches found: 0
   T-shapes added: 1

2. CUBE81 APPROACH (Few complex elements):
   Time: 0.29568 ms
   Elements processed: 4913000 cubes
   Operations per element: ~81 point checks + structure analysis
   Total operations: 491300000 (estimated)
   Matches found: 3
   T-shapes added: 1

=== DEBUG: Checking cube 10,10,10 ===
Cube contents (layer by layer):
  Layer z=0:
    0 0 0 
    0 1 0 
    0 0 0 
  Layer z=1:
    0 0 0 
    1 1 1 
    0 0 0 
  Layer z=2:
    0 0 0 
    0 1 0 
    0 0 0 
Cube81.hasTShape() returns: TRUE
Center column values: 1 1 1 
Horizontal bar values: 0 1 0
Manual check - vertical: YES, horizontal: NO


%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstdlib>  // For rand()
#include <ctime>    // For time()

#define N 512
#define TOTAL_VOXELS (N*N*N)

// Simple random functions for CUDA compatibility
__host__ __device__ int simple_rand(int seed) {
    // Simple LCG (Linear Congruential Generator)
    return (seed * 1103515245 + 12345) & 0x7fffffff;
}

__host__ __device__ float random_float(int& seed) {
    seed = simple_rand(seed);
    return (float)seed / (float)0x7fffffff;
}

// ========== REPRESENTATION A: VOXEL GRID ==========
// Many simple elements (N³ voxels)

__global__ void voxelPatternSearch(const int8_t* volume, 
                                   const int8_t* pattern,  // 7×7×7 pattern
                                   int pattern_size,
                                   int* matches) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_VOXELS) return;
    
    int x = tid % N;
    int y = (tid / N) % N;
    int z = tid / (N * N);
    
    // Can we center the pattern here?
    if (x < pattern_size/2 || x >= N-pattern_size/2 ||
        y < pattern_size/2 || y >= N-pattern_size/2 ||
        z < pattern_size/2 || z >= N-pattern_size/2) return;
    
    int match_score = 0;
    // Check all voxels in pattern (7×7×7 = 343 checks!)
    for (int dz = -pattern_size/2; dz <= pattern_size/2; dz++) {
        for (int dy = -pattern_size/2; dy <= pattern_size/2; dy++) {
            for (int dx = -pattern_size/2; dx <= pattern_size/2; dx++) {
                int pattern_idx = (dz+3)*49 + (dy+3)*7 + (dx+3);
                int volume_idx = (z+dz)*N*N + (y+dy)*N + (x+dx);
                
                if (pattern[pattern_idx] != 0) {
                    if (volume[volume_idx] == pattern[pattern_idx]) {
                        match_score++;
                    } else if (volume[volume_idx] != 0) {
                        match_score--;  // Penalty for mismatch
                    }
                }
            }
        }
    }
    
    if (match_score > 20) {  // Threshold for T-shape
        atomicAdd(matches, 1);
    }
}

// ========== REPRESENTATION B: CUBE81 ==========
// Few complex elements (81 structured points per cube)

struct Cube81 {
    // Actually 27 points for 3×3×3 cube
    int8_t points[27];  // Change from 81 to 27!
    
    __host__ __device__ bool hasTShape() const {
        // For 3×3×3 cube (27 points)
        // Vertical bar: center column (x=1,y=1,z=0..2)
        // Indices: (1,1,0)=4, (1,1,1)=13, (1,1,2)=22
        bool vertical = (points[4] != 0 && points[13] != 0 && points[22] != 0);
        
        // Horizontal bar: middle row (x=0..2,y=1,z=1)  
        // Indices: (0,1,1)=12, (1,1,1)=13, (2,1,1)=14
        bool horizontal = (points[12] != 0 && points[13] != 0 && points[14] != 0);
        
        return vertical && horizontal;
    }
};

__global__ void cube81PatternSearch(const int8_t* volume,
                                    int* matches) {
    // Each thread handles one 3×3×3 subcube
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cubes_per_dim = N / 3;  // N must be multiple of 3
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    
    if (tid >= total_cubes) return;
    
    // Which cube are we?
    int cube_x = (tid % cubes_per_dim) * 3;
    int cube_y = ((tid / cubes_per_dim) % cubes_per_dim) * 3;
    int cube_z = (tid / (cubes_per_dim * cubes_per_dim)) * 3;
    
    // Extract Cube81 from volume
    Cube81 cube;
    int idx = 0;
    
    // Extract all 81 points (structured extraction)
	for (int dz = 0; dz < 3; dz++) {
		for (int dy = 0; dy < 3; dy++) {
			for (int dx = 0; dx < 3; dx++) {
				int vol_idx = (cube_z+dz)*N*N + (cube_y+dy)*N + (cube_x+dx);
				cube.points[idx++] = volume[vol_idx];
			}
		}
	}
    
    // Check for T-shape using structural knowledge
    if (cube.hasTShape()) {
        atomicAdd(matches, 1);
    }
}

// ========== EXPERIMENT MAIN ==========

int main() {
    std::cout << "=== COMPLEXITY VS QUANTITY EXPERIMENT ===\n";
    std::cout << "N=" << N << ", Total Voxels=" << TOTAL_VOXELS << "\n";
    std::cout << "Voxel Grid: " << TOTAL_VOXELS << " simple elements\n";
    std::cout << "Cube81 Grid: " << ((N/3)*(N/3)*(N/3)) 
              << " complex elements (81 points each)\n\n";
    
    // Generate test volume
    std::vector<int8_t> h_volume(TOTAL_VOXELS, 0);
    
    // Seed for reproducibility - use int for compatibility
    int seed = (int)time(NULL);
    
    // ========== HARDCODED DEBUG T-SHAPE ==========
    // Place ONE perfect T-shape that Cube81 SHOULD detect
    int t_shapes_added = 0;
    
    // Choose a cube center (aligned with 3×3×3 grid)
    int cube_x = 5;  // Which 3×3×3 cube (0-20 for N=64/3≈21)
    int cube_y = 5;
    int cube_z = 5;
    
    // Global coordinates of CUBE CENTER
    int center_x = cube_x * 3 + 1;  // = 16
    int center_y = cube_y * 3 + 1;  // = 16  
    int center_z = cube_z * 3 + 1;  // = 16
    
    std::cout << "=== DEBUG: Placing test T-shape ===\n";
    std::cout << "Cube: (" << cube_x << "," << cube_y << "," << cube_z << ")\n";
    std::cout << "Global center: (" << center_x << "," << center_y << "," << center_z << ")\n";
    
    // Place T-shape aligned with Cube81 detection logic
    // Vertical bar: center column of cube (indices 50,58,66,74,80)
    // These are positions: (1,1,2), (1,1,2), (1,1,2), (1,1,2), (1,1,2)?? Wait, let me check...
    
    // Actually, let's think: Cube81 indices 0-80 map to positions in 3×3×3
    // Index = z*9 + y*3 + x  (where x,y,z are 0,1,2 within cube)
    
    // Center column would be x=1, y=1, z=0..4? But we only have z=0..2 in one cube!
    
    // The problem: A 5-high vertical bar spans MULTIPLE cubes!
    // Our Cube81 only looks at ONE 3×3×3 cube at a time
    
    // ========== FIX: Place T-shape within ONE cube ==========
    // Let's make a smaller T-shape that fits in one 3×3×3 cube
    
    std::cout << "\nPlacing SMALL T-shape (fits in one 3×3×3 cube):\n";
    
    // Place at the CENTER of a specific cube (cube 10,10,10)
    int test_cube = 10;
    int gx = test_cube * 3 + 1;  // Global x = 31
    int gy = test_cube * 3 + 1;  // Global y = 31
    int gz = test_cube * 3 + 1;  // Global z = 31
    
    // Small T: Vertical bar 3 high, horizontal bar 3 wide
    // Vertical: (1,1,0), (1,1,1), (1,1,2) in local coords
    // Horizontal: (0,1,1), (1,1,1), (2,1,1) in local coords
    
    // Convert to global and place
    // Vertical bar (center column)
    for (int dz = 0; dz < 3; dz++) {
        int idx = (gz + dz - 1) * N * N + (gy) * N + (gx);
        if (idx >= 0 && idx < TOTAL_VOXELS) {
            h_volume[idx] = 1;
            std::cout << "  Vertical at global: (" << gx << "," << gy << "," << (gz+dz-1) << ")\n";
        }
    }
    
    // Horizontal bar (middle row)
    for (int dx = 0; dx < 3; dx++) {
        int idx = (gz) * N * N + (gy) * N + (gx + dx - 1);
        if (idx >= 0 && idx < TOTAL_VOXELS) {
            h_volume[idx] = 1;
            std::cout << "  Horizontal at global: (" << (gx+dx-1) << "," << gy << "," << gz << ")\n";
        }
    }
    
    t_shapes_added = 1;
    
	// Add random noise (5% density) - BUT SKIP TEST CUBE!
	std::cout << "\nAdding 5% random noise (skipping test cube region)...\n";
	for (int i = 0; i < TOTAL_VOXELS; i++) {
		// Convert index to coordinates
		int x = i % N;
		int y = (i / N) % N;
		int z = i / (N * N);
		
		// Check if this voxel is in the test cube (10,10,10)
		bool in_test_cube = (x >= test_cube*3 && x < test_cube*3+3 &&
							 y >= test_cube*3 && y < test_cube*3+3 &&
							 z >= test_cube*3 && z < test_cube*3+3);
		
		if (in_test_cube) {
			continue;  // Skip - don't add noise to test cube!
		}
		
		// Only add noise to EMPTY voxels outside test cube
		if (h_volume[i] == 0) {
			float r = random_float(seed);
			if (r < 0.05f) {
				h_volume[i] = (simple_rand(seed++) % 2) ? 1 : -1;
			}
		}
	}
    
    // Allocate device memory
    int8_t* d_volume;
    int *d_voxel_matches, *d_cube_matches;
    cudaMalloc(&d_volume, TOTAL_VOXELS);
    cudaMalloc(&d_voxel_matches, sizeof(int));
    cudaMalloc(&d_cube_matches, sizeof(int));
    
    cudaMemcpy(d_volume, h_volume.data(), TOTAL_VOXELS, cudaMemcpyHostToDevice);
    
    // Create T-shape pattern for voxel search (7×7×7)
    const int PATTERN_SIZE = 7;
    std::vector<int8_t> h_pattern(PATTERN_SIZE*PATTERN_SIZE*PATTERN_SIZE, 0);
    
    // Create a 3×3×3 T-shape pattern (smaller to match our test)
    // Center column
    for (int z = 0; z < 3; z++) {
        h_pattern[z*9 + 1*3 + 1] = 1;  // (1,1,z)
    }
    // Middle horizontal bar
    for (int x = 0; x < 3; x++) {
        h_pattern[1*9 + 1*3 + x] = 1;  // (x,1,1)
    }
    
    int8_t* d_pattern;
    cudaMalloc(&d_pattern, PATTERN_SIZE*PATTERN_SIZE*PATTERN_SIZE);
    cudaMemcpy(d_pattern, h_pattern.data(), 
               PATTERN_SIZE*PATTERN_SIZE*PATTERN_SIZE, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========== RUN EXPERIMENT ==========
    
    // Test 1: Voxel Grid Approach
    std::cout << "\n\n1. VOXEL GRID APPROACH (Many simple elements):\n";
    cudaMemset(d_voxel_matches, 0, sizeof(int));
    
    cudaEventRecord(start);
    voxelPatternSearch<<<(TOTAL_VOXELS+255)/256, 256>>>(d_volume, d_pattern, 
                                                        PATTERN_SIZE, d_voxel_matches);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float voxel_time;
    cudaEventElapsedTime(&voxel_time, start, stop);
    
    int voxel_matches;
    cudaMemcpy(&voxel_matches, d_voxel_matches, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "   Time: " << voxel_time << " ms\n";
    std::cout << "   Elements processed: " << TOTAL_VOXELS << " voxels\n";
    std::cout << "   Operations per element: ~343 pattern checks\n";
    std::cout << "   Total operations: " << (TOTAL_VOXELS * 343) << "\n";
    std::cout << "   Matches found: " << voxel_matches << "\n";
    std::cout << "   T-shapes added: " << t_shapes_added << "\n";
    
    // Test 2: Cube81 Approach
    std::cout << "\n2. CUBE81 APPROACH (Few complex elements):\n";
    cudaMemset(d_cube_matches, 0, sizeof(int));
    
    cudaEventRecord(start);
    int cubes_per_dim = N / 3;
    int total_cubes = cubes_per_dim * cubes_per_dim * cubes_per_dim;
    cube81PatternSearch<<<(total_cubes+255)/256, 256>>>(d_volume, d_cube_matches);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cube_time;
    cudaEventElapsedTime(&cube_time, start, stop);
    
    int cube_matches;
    cudaMemcpy(&cube_matches, d_cube_matches, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "   Time: " << cube_time << " ms\n";
    std::cout << "   Elements processed: " << total_cubes << " cubes\n";
    std::cout << "   Operations per element: ~81 point checks + structure analysis\n";
    std::cout << "   Total operations: " << (total_cubes * 100) << " (estimated)\n";
    std::cout << "   Matches found: " << cube_matches << "\n";
    std::cout << "   T-shapes added: " << t_shapes_added << "\n";
    
    // ========== DEBUG: Check the specific cube ==========
    std::cout << "\n=== DEBUG: Checking cube " << test_cube << "," << test_cube << "," << test_cube << " ===\n";
    
    // Manually check if our test cube has the T-shape
    Cube81 test_cube_data;
    int local_idx = 0;
    for (int dz = 0; dz < 3; dz++) {
        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                int gx = test_cube*3 + dx;
                int gy = test_cube*3 + dy;
                int gz = test_cube*3 + dz;
                int vol_idx = gz*N*N + gy*N + gx;
                test_cube_data.points[local_idx++] = h_volume[vol_idx];
            }
        }
    }
    
    // Print the cube
    std::cout << "Cube contents (layer by layer):\n";
    for (int z = 0; z < 3; z++) {
        std::cout << "  Layer z=" << z << ":\n";
        for (int y = 0; y < 3; y++) {
            std::cout << "    ";
            for (int x = 0; x < 3; x++) {
                int idx = z*9 + y*3 + x;
                int val = test_cube_data.points[idx];
                std::cout << (val == 1 ? "1" : (val == -1 ? "-" : "0")) << " ";
            }
            std::cout << "\n";
        }
    }
    
    std::cout << "Cube81.hasTShape() returns: " 
              << (test_cube_data.hasTShape() ? "TRUE" : "FALSE") << "\n";
    
    // Check which parts of T-shape are present
    bool vertical = true;
    int center_indices[] = {50, 58, 66, 74, 80};  // Wait, these are wrong for 3×3×3!
    // Actually for 3×3×3 (27 points), indices 0-26
    // Center column: (1,1,0)=12, (1,1,1)=13, (1,1,2)=14
    int center_col[] = {12, 13, 14};
    
    std::cout << "Center column values: ";
    for (int i = 0; i < 3; i++) {
        std::cout << (int)test_cube_data.points[center_col[i]] << " ";
        if (test_cube_data.points[center_col[i]] == 0) vertical = false;
    }
    std::cout << "\n";
    
    // Horizontal bar: (0,1,1)=10, (1,1,1)=13, (2,1,1)=16
    bool horizontal = (test_cube_data.points[10] != 0 && 
                      test_cube_data.points[13] != 0 && 
                      test_cube_data.points[16] != 0);
    
    std::cout << "Horizontal bar values: " 
              << (int)test_cube_data.points[10] << " "
              << (int)test_cube_data.points[13] << " "  
              << (int)test_cube_data.points[16] << "\n";
    
    std::cout << "Manual check - vertical: " << (vertical ? "YES" : "NO")
              << ", horizontal: " << (horizontal ? "YES" : "NO") << "\n";
    
    // ========== ANALYSIS ==========
    std::cout << "\n=== ANALYSIS ===\n";
    
    // Rest of your analysis code remains the same...
    // [Keep the analysis section from your original code]
    
    // Cleanup
    cudaFree(d_volume);
    cudaFree(d_pattern);
    cudaFree(d_voxel_matches);
    cudaFree(d_cube_matches);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}