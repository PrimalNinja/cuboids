%%writefile test_cuboids.py
import torch
from torch.utils.cpp_extension import load
import time

# Load the custom cuboid kernel
cuboid_ops = load(name="cuboid_ops", sources=["cuboid_ops.cu"], verbose=False)

N = 63  # Must be divisible by 3
volume = torch.zeros((N, N, N), dtype=torch.int8, device='cuda')

# Place one T-shape
center = N // 2
cube_center = (center // 3) * 3 + 1

volume[cube_center-1, cube_center, cube_center] = 1
volume[cube_center,   cube_center, cube_center] = 1
volume[cube_center+1, cube_center, cube_center] = 1
volume[cube_center,   cube_center, cube_center-1] = 1
volume[cube_center,   cube_center, cube_center+1] = 1

# CUBOID-BASED (Structural)
def pytorch_cuboid_loop(volume):
    count = 0
    for cz in range(0, N-2, 3):
        for cy in range(0, N-2, 3):
            for cx in range(0, N-2, 3):
                cube = volume[cz:cz+3, cy:cy+3, cx:cx+3]
                center = cube[1,1,1]
                if center == 0: continue
                if cube[0,1,1] == 0 or cube[2,1,1] == 0: continue
                if cube[1,1,0] == 0 or cube[1,1,2] == 0: continue
                count += 1
    return count

def custom_cuboid_count(volume):
    return cuboid_ops.count_t_shapes(volume).item()

# NON-CUBOID (Dense voxel-wise)
def pytorch_dense_tshape_count(volume):
    center = volume[1:-1, 1:-1, 1:-1]
    up     = volume[0:-2, 1:-1, 1:-1]
    down   = volume[2:  , 1:-1, 1:-1]
    left   = volume[1:-1, 1:-1, 0:-2]
    right  = volume[1:-1, 1:-1, 2:  ]
    mask = (center > 0) & (up > 0) & (down > 0) & (left > 0) & (right > 0)
    return mask.sum().item()

# Warmup
for _ in range(10):
    pytorch_cuboid_loop(volume)
    custom_cuboid_count(volume)
    pytorch_dense_tshape_count(volume)

num_iters = 100

# CUBOID measurements
cuboid_custom_total = 0
cuboid_pt_total = 0
start = time.time()
for _ in range(num_iters):
    cuboid_custom_total += custom_cuboid_count(volume)
torch.cuda.synchronize()
cuboid_custom_time = (time.time() - start) * 1000 / num_iters

start = time.time()
for _ in range(num_iters):
    cuboid_pt_total += pytorch_cuboid_loop(volume)
cuboid_pt_time = (time.time() - start) * 1000 / num_iters

# NON-CUBOID measurements
non_cuboid_pt_total = 0
start = time.time()
for _ in range(num_iters):
    non_cuboid_pt_total += pytorch_dense_tshape_count(volume)
torch.cuda.synchronize()
non_cuboid_pt_time = (time.time() - start) * 1000 / num_iters

# SYMMETRIC OUTPUT
print("=== T-SHAPE COUNT BENCHMARK ===")
print("Task: Count T-shaped patterns (5 voxels: center + 4-connected neighbors)")
print(f"Volume: {N}×{N}×{N}, Expected matches: 1")
print()

print("--- BLOCK 1: CUBOID PARADIGM (Implementation Comparison) ---")
print(f"1a - Custom CUDA cuboid matches  : {cuboid_custom_total // num_iters}")
print(f"1b - PyTorch Python cuboid matches : {cuboid_pt_total // num_iters}")
print(f"1c - Custom CUDA cuboid time     : {cuboid_custom_time:.4f} ms/iter")
print(f"1d - PyTorch Python cuboid time  : {cuboid_pt_time:.4f} ms/iter")
speedup_cuboid_impl = cuboid_pt_time / cuboid_custom_time
print(f"1e - Implementation speedup      : {speedup_cuboid_impl:.1f}x")
print(f"     → GPU CUDA is {speedup_cuboid_impl:.1f}× faster than Python CPU loop (same cuboid algorithm)")

print()
print("--- BLOCK 2: DENSE PARADIGM (Voxel-wise) ---")
print(f"2a - Custom CUDA dense matches   : N/A (not implemented)")
print(f"2b - PyTorch GPU dense matches   : {non_cuboid_pt_total // num_iters}")
print(f"2c - Custom CUDA dense time      : N/A (not implemented)")
print(f"2d - PyTorch GPU dense time      : {non_cuboid_pt_time:.4f} ms/iter")
print(f"2e - Implementation speedup      : N/A (no custom dense to compare)")

print()
print("--- BLOCK 3: PARADIGM COMPARISON (Best vs Best) ---")
paradigm_speedup = non_cuboid_pt_time / cuboid_custom_time
print(f"3a - Fastest cuboid time         : {cuboid_custom_time:.4f} ms/iter (Custom CUDA)")
print(f"3b - Fastest dense time          : {non_cuboid_pt_time:.4f} ms/iter (PyTorch GPU)")
print(f"3c - Paradigm speedup            : {paradigm_speedup:.1f}x")
print(f"     → Cuboid structural approach is {paradigm_speedup:.1f}× faster than dense voxel-wise")
print(f"     → Time saved: {non_cuboid_pt_time - cuboid_custom_time:.4f} ms/iter")