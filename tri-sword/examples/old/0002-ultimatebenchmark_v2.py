%%writefile ultimate_benchmark_v2.py
import torch
import numpy as np
import time
from torch.utils.cpp_extension import load

print("🗡️ TRI-SWORD V2: MEMORY ALIGNMENT INVESTIGATION 🗡️")
print("Compiling CUDA kernels...")

benchmark_ops = load(
    name="benchmark_ops_v2",
    sources=["ultimate_benchmark_v2.cu"],
    verbose=True,
    extra_cuda_cflags=['-O3']
)

print("✓ Kernels compiled!\n")

BATCH_SIZE = 1000
FREQ_BINS = 128
TIME_SLICES = 2
NUM_RUNS = 100

def generate_test_data(batch_size, num_kicks=50):
    spectrograms = torch.rand(batch_size, FREQ_BINS * TIME_SLICES, device='cuda') * 0.5
    kick_indices = np.random.choice(batch_size, num_kicks, replace=False)
    
    for idx in kick_indices:
        spectrograms[idx, :20] = torch.rand(20, device='cuda') * 3.0 + 2.0
        spectrograms[idx, FREQ_BINS:FREQ_BINS+20] = torch.rand(20, device='cuda') * 3.0 + 2.0
        spectrograms[idx, 60:FREQ_BINS] *= 0.3
        spectrograms[idx, FREQ_BINS+60:] *= 0.3
    
    return spectrograms, set(kick_indices)

test_data, ground_truth = generate_test_data(BATCH_SIZE)

def benchmark(name, func, data, runs=NUM_RUNS):
    print(f"Testing: {name}")
    
    for _ in range(10):
        func(data)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(runs):
        start = time.time()
        result = func(data)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    detections = result.item()
    
    print(f"  Time: {avg_time:.4f} ms | Detections: {detections}")
    return avg_time, detections

print("=" * 70)
print("🔬 MEMORY ALIGNMENT INVESTIGATION")
print("=" * 70)
print()

results = {}

results['Cube27 (27B)'] = benchmark(
    "Cube27 (27 bytes, 32-aligned)",
    lambda x: benchmark_ops.approach9_optimized_cube27(x),
    test_data
)

results['Cube81 (81B)'] = benchmark(
    "Cube81 (81 bytes, 128-aligned)",
    lambda x: benchmark_ops.approach8_cube81(x),
    test_data
)

results['Coalesced27'] = benchmark(
    "Coalesced Cube27 (warp-optimized)",
    lambda x: benchmark_ops.approach10_coalesced(x),
    test_data
)

print("\n" + "=" * 70)
print("📊 ANALYSIS: WHY DOES CUBE81 WORK SO WELL?")
print("=" * 70)

for name, (time, _) in results.items():
    print(f"{name:30} {time:.4f} ms")

fastest = min(results.values(), key=lambda x: x[0])
fastest_name = [k for k, v in results.items() if v[0] == fastest[0]][0]

print(f"\n🏆 Winner: {fastest_name}")
print("\n💡 THEORY:")
print("  Cube27 (27B) → 32-byte aligned = 1 cache sector")
print("  Cube81 (81B) → 128-byte aligned = 1 cache LINE")
print("  • Cube81 = 3 layers of 27 points each")
print("  • Perfect for GPU cache hierarchy!")
print("  • Warp of 32 threads × 81 bytes = 2592 bytes")
print("  • Fits in L1 cache (48KB) perfectly!")