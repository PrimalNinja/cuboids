%%writefile testharness.py
import torch
import numpy as np
import sys
import os
from torch.utils.cpp_extension import load

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Sync for accurate traceback

print("JIT compiling Ternary Maze Solver...")
tri_sword_cuda = load(
    name='tri_sword_cuda',
    sources=['tri_sword.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class TriSwordBlade:
    def __init__(self, max_batch):
        self.handle = tri_sword_cuda.sharpen(max_batch, 256)

    def strike(self, dim, data):
        return tri_sword_cuda.slash(self.handle, 'D', 'T', dim, data)

    def __del__(self):
        tri_sword_cuda.sheath(self.handle)

def run_benchmark(dim, batch_size=1000000):
    """Run maze solving benchmark for given dimension"""
    grid_size = dim * dim

    # Allocate properly sized tensor for this dimension
    maze_universe = torch.zeros((batch_size, grid_size), dtype=torch.int8, device='cuda')

    # Initialize: frontier at position [0,0] (index 0)
    maze_universe[:, 0] = -1

    blade = TriSwordBlade(batch_size)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    blade.strike(dim, maze_universe)
    end.record()
    torch.cuda.synchronize()

    total_time_ms = start.elapsed_time(end)
    time_per_maze_us = (total_time_ms * 1000) / batch_size

    return total_time_ms, time_per_maze_us

def main():
    BASE_BATCH = 1000000
    VRAM_CAP_GB = 16  # Adjust to your GPU (e.g., 15 for T4, 24 for 3090)

    print(f"\n--- TERNARY MAZE SOLVER AUDIT (Base Batch: {BASE_BATCH}) ---")
    print(f"{'Dim':<6} | {'Grid Cells':<12} | {'Batch':<10} | {'Total (ms)':<12} | {'Per Maze (µs)':<15} | {'Cells/µs':<12}")
    print("-" * 90)

    # Test dimensions - adjusted for memory constraints
    dimensions = [10, 50, 100, 200]

    for dim in dimensions:
        grid_size = dim * dim
        memory_needed_gb = (BASE_BATCH * grid_size) / (1024**3)
        batch_size = BASE_BATCH
        scale_msg = ""

        if memory_needed_gb > VRAM_CAP_GB:
            # Halve (or more) until fits
            original_batch = batch_size
            while memory_needed_gb > VRAM_CAP_GB and batch_size > 1:
                batch_size //= 2
                memory_needed_gb = (batch_size * grid_size) / (1024**3)
            if batch_size < original_batch:
                scale_msg = f"AUTO-HALVED to {batch_size:,} "
            if batch_size == 1:
                print(f"{dim:<6} | {grid_size:<12} | {scale_msg}SKIPPED - still >{VRAM_CAP_GB}GB")
                continue

        try:
            total_ms, per_maze_us = run_benchmark(dim, batch_size)
            cells_per_us = grid_size / per_maze_us if per_maze_us > 0 else 0

            batch_str = f"{batch_size:,}" if batch_size != BASE_BATCH else f"{batch_size:,} (full)"
            print(f"{dim:<6} | {grid_size:<12} | {batch_str:<10} | {total_ms:>10.2f} | {per_maze_us:>13.4f} | {cells_per_us:>10.2f}")

        except RuntimeError as e:
            print(f"{dim:<6} | {grid_size:<12} | {scale_msg}ERROR: {str(e)[:40]}")

        # Clear memory
        torch.cuda.empty_cache()

    print(f"\n{'='*90}")
    print(f"Brute Force Without Repetition: Sovereign Ternary Engine Active.")
    print(f"{'='*90}")

    # Optional: Single large maze test with smaller batch
    print(f"\n--- LARGE MAZE TEST (Smaller Batch) ---")
    large_dims = [500, 1000]
    small_batch = 10000  # Reduce batch for memory

    for dim in large_dims:
        grid_size = dim * dim
        memory_needed_gb = (small_batch * grid_size) / (1024**3)

        if memory_needed_gb > VRAM_CAP_GB:
            print(f"{dim}×{dim}: SKIPPED - needs {memory_needed_gb:.1f}GB VRAM")
            continue

        # Patch: For warp test, use batch=1 for single maze on large dims
        test_batch = 1 if dim >= 500 else small_batch
        try:
            total_ms, per_maze_us = run_benchmark(dim, test_batch)
            cells_per_us = grid_size / per_maze_us if per_maze_us > 0 else 0

            print(f"{dim}×{dim} maze ({grid_size:,} cells) with {test_batch:,} mazes (warp mode: {test_batch==1}):")
            print(f"  Total: {total_ms:.2f}ms | Per maze: {per_maze_us:.4f}µs | Throughput: {cells_per_us:.0f} cells/µs")

        except RuntimeError as e:
            print(f"{dim}×{dim}: ERROR - {str(e)[:60]}")

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()