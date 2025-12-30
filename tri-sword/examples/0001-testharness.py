"""
Tri-Sword Test Harness: Discovery One Collision Detection
Realistic scenario: Multiple Discovery One ships rotating in 3D space
"""

import importlib
import torch
import time
import math
from torch.utils.cpp_extension import load

print("JIT compiling tri_sword_cuda...")
tri_sword_cuda = load(
    name='tri_sword_cuda',
    sources=['tri_sword.cu'],
    extra_cflags=['-O2'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)
print("Compilation complete!")

TESTNUMBER       = 1
TESTDESCRIPTION  = "Discovery One fleet collision detection"
BASELINE_MODULE  = 'discoveryone_detection'
FUNCTION_NAME    = 'discoveryone_detection'
SHAPE            = 'C'  # Cuboid
DATATYPE         = 'T'  # Trit (ternary: 0, 1, 2)
ALGORITHM        = 'T'  # Threaded
PERMUTATION      = 5    # F + C (Faces + Cells)

baseline_module = importlib.import_module(BASELINE_MODULE)
baseline_function = getattr(baseline_module, FUNCTION_NAME)

def generate_discovery_fleet(N=30, num_ships=50):
    """
    Generate a 3D voxel grid with Discovery One ships
    
    Args:
        N: Grid dimension (NxNxN)
        num_ships: Number of Discovery One ships to place
    
    Returns:
        Tensor of shape (N, N, N) with ships marked
    """
    grid = torch.zeros(N, N, N)
    
    # Each Discovery One is a linear structure (elongated ship)
    # Simplified: 10 voxels in a line with random rotation
    
    for ship_id in range(num_ships):
        # Random position in space
        x = torch.randint(5, N-5, (1,)).item()
        y = torch.randint(5, N-5, (1,)).item()
        z = torch.randint(5, N-5, (1,)).item()
        
        # Random rotation (simplified to axis-aligned for now)
        axis = torch.randint(0, 3, (1,)).item()  # 0=x, 1=y, 2=z
        
        # Place ship as line of voxels
        for i in range(10):
            if axis == 0:  # Along x-axis
                xi, yi, zi = x + i - 5, y, z
            elif axis == 1:  # Along y-axis
                xi, yi, zi = x, y + i - 5, z
            else:  # Along z-axis
                xi, yi, zi = x, y, z + i - 5
            
            # Bounds check
            if 0 <= xi < N and 0 <= yi < N and 0 <= zi < N:
                grid[xi, yi, zi] += 1.0  # Accumulate for collision detection
    
    return grid

def run_baseline(data):
    start = time.time()
    result = baseline_function(data)
    end = time.time()
    elapsed = end - start
    return result, elapsed

def run_cuda_kernel(func_name, shape, datatype, algorithm, permutation, data):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    result = tri_sword_cuda.slash(func_name, shape, datatype, algorithm, permutation, data)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1000.0
    return elapsed, result

def main():
    # Generate realistic Discovery One fleet scenario
    N = 30  # 30x30x30 space
    num_ships = 50  # 50 Discovery One ships
    
    print(f"=== TEST {TESTNUMBER}: {TESTDESCRIPTION} ===")
    print(f"Space: {N}x{N}x{N} = {N**3:,} voxels")
    print(f"Fleet: {num_ships} Discovery One ships")
    print(f"Expected occupancy: ~{num_ships * 10} voxels (~{(num_ships*10)/(N**3)*100:.1f}%)")
    print()

    data = generate_discovery_fleet(N=N, num_ships=num_ships)
    data = data.cuda()

    baseline_result, baseline_time = run_baseline(data)
    print(f"[Baseline] {FUNCTION_NAME} time: {baseline_time:.6f} sec")
    print(f"[Baseline] Ships detected: {baseline_result[0].item():.1f}")
    print(f"[Baseline] Collisions: {baseline_result[1].item():.0f}")
    print()

    cuda_time, cuda_result = run_cuda_kernel(FUNCTION_NAME, SHAPE, DATATYPE, ALGORITHM, PERMUTATION, data)
    print(f"[CUDA] {SHAPE}{DATATYPE}{ALGORITHM}{PERMUTATION} time: {cuda_time:.6f} sec")
    print(f"[CUDA] Output size: {len(cuda_result)} elements")
    print(f"[CUDA] First values: {cuda_result[:10].cpu().numpy()}")
    print()
    
    print(f"Completed: {FUNCTION_NAME} | {SHAPE}{DATATYPE}{ALGORITHM}{PERMUTATION}")
    print(f"Speedup: {baseline_time/cuda_time:.2f}x")
    print("="*50)

if __name__ == "__main__":
    main()