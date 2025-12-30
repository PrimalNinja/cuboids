%%writefile testharness.py
"""
Tri-Sword Test Harness: Audio Beat Detection
Testing Vector vs Triangle vs Pyramid vs Cuboid vs Nonoid
"""

import importlib
import torch
import numpy as np
import time
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

TESTNUMBER = 1
TESTDESCRIPTION = "Audio kick drum detection"
BASELINE_MODULE = 'beat_detection'
FUNCTION_NAME = 'beat_detection'

# Mocking the baseline for standalone runs if module not found
try:
    baseline_module = importlib.import_module(BASELINE_MODULE)
    baseline_function = getattr(baseline_module, FUNCTION_NAME)
except ImportError:
    def baseline_function(data):
        # Fallback simulation of detection logic
        b = data[:, :20].sum(dim=1) + data[:, 128:148].sum(dim=1)
        h = data[:, 60:128].sum(dim=1) + data[:, 188:256].sum(dim=1)
        return ((b > 39.95) & (h < 20.05)).sum()

def generate_audio_spectrograms(batch_size=50000, freq_bins=128, time_slices=2, num_kicks=500):
    spectrograms = torch.rand(batch_size, freq_bins * time_slices, device='cuda') * 0.5
    kick_indices = np.random.choice(batch_size, num_kicks, replace=False)
    
    for idx in kick_indices:
        spectrograms[idx, :20] = torch.rand(20, device='cuda') * 3.0 + 2.0
        spectrograms[idx, freq_bins:freq_bins+20] = torch.rand(20, device='cuda') * 3.0 + 2.0
        spectrograms[idx, 60:freq_bins] *= 0.3
        spectrograms[idx, freq_bins+60:] *= 0.3
    
    return spectrograms, set(kick_indices)

def run_baseline(data):
    start = time.time()
    result = baseline_function(data)
    elapsed = time.time() - start
    return result, elapsed

def run_typical_cuda(func_name, points, data):  # Add points here
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # Ensure points is passed to match C++ signature: typical(string, int, tensor)
    result = tri_sword_cuda.typical(func_name, points, data) 
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1000.0
    return elapsed, result

def run_cuda_kernel(func_name, shape, datatype, algorithm, points, data):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = tri_sword_cuda.slash(func_name, shape, datatype, algorithm, points, data)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1000.0
    return elapsed, result

def main():
    BATCH_SIZE = 50000
    FREQ_BINS = 128
    TIME_SLICES = 2
    NUM_KICKS = 500
    
    print(f"\n{'='*70}")
    print(f"TEST {TESTNUMBER}: {TESTDESCRIPTION}")
    print(f"{'='*70}")
    print(f"Batch: {BATCH_SIZE} spectrograms")
    
    data, ground_truth = generate_audio_spectrograms(BATCH_SIZE, FREQ_BINS, TIME_SLICES, NUM_KICKS)
    data = data.contiguous()

    # 1. Run Baseline
    baseline_result, baseline_time = run_baseline(data)
    
    # NEW: WARM-UP PASS (The "Sacrificial" Run)
    # We run the kernel once and discard the timing to 'prime' the GPU
    print("Warming up GPU kernels...")
    _ = tri_sword_cuda.typical(FUNCTION_NAME, 40, data)
    _ = tri_sword_cuda.slash(FUNCTION_NAME, 'V', 'B', 'T', 40, data)
    torch.cuda.synchronize()
    print("Warm-up complete. Starting real tests.\n")

    # Now the REAL timing starts
    typical_time, typical_result = run_typical_cuda(FUNCTION_NAME, 512, data)
    typical_kicks = typical_result[0].item()
    
    print(f"[Typical] CUDA Optimized")
    print(f"  Time: {typical_time*1000:.3f} ms\n")

    # 3. Setup Results List & Add Typical Baseline First
    results = []
    results.append({
        'label': 'Typical (Baseline)',
        'code': 'TYP512',
        'time': typical_time,
        'kicks': typical_kicks,
        'points': 512
    })
    
    # 4. Expert Configurations
    configs = [
        ('V', 'B', 'T', 40,  'VBT40'),
        ('V', 'T', 'T', 40,  'VTT40'),
        ('T', 'B', 'T', 40,  'TBT40'),
        ('T', 'T', 'T', 40,  'TTT40'),
        ('P', 'B', 'T', 40,  'PBT40'),
        ('P', 'T', 'T', 40,  'PTT40'),
        ('C', 'B', 'T', 80,  'CBT80'),
        ('C', 'T', 'T', 80,  'CTT80'),
        ('N', 'B', 'T', 111, 'NBT111'),
        ('N', 'T', 'T', 111, 'NTT111'),

        ('V', 'B', 'T', 512,  'VBT512'),
        ('V', 'T', 'T', 512,  'VTT512'),
        ('T', 'B', 'T', 512,  'TBT512'),
        ('T', 'T', 'T', 512,  'TTT512'),
        ('P', 'B', 'T', 512,  'PBT512'),
        ('P', 'T', 'T', 512,  'PTT512'),
        ('C', 'B', 'T', 512,  'CBT512'),
        ('C', 'T', 'T', 512,  'CTT512'),
        ('N', 'B', 'T', 512, 'NBT512'),
        ('N', 'T', 'T', 512, 'NTT512'),

        ('N', 'T', 'T', 1024, 'NTT1024'),
        ('N', 'T', 'T', 2048, 'NTT2048'),
        ('N', 'T', 'T', 4096, 'NTT4096'),
        ('N', 'T', 'T', 8192, 'NTT8192'),
        ('N', 'T', 'T', 16384, 'NTT16384'),
        ('N', 'T', 'T', 32768, 'NTT32768')
    ]
    
    for i, (shape, dtype, algo, points, label) in enumerate(configs, 1):
        # Prepare 8-bit data
        if dtype == 'B':
            input_data = (data * 20.0).to(torch.uint8).contiguous()
        else:
            input_data = (data * 20.0).to(torch.int8).contiguous()

        cuda_time, cuda_result = run_cuda_kernel(FUNCTION_NAME, shape, dtype, algo, points, input_data)
        cuda_kicks = cuda_result[0].item()
        
        code = f"{shape}{dtype}{algo}{points}"
        results.append({
            'label': label,
            'code': code,
            'time': cuda_time,
            'kicks': cuda_kicks,
            'points': points  # <--- ADD THIS LINE
        })
    
        # 1. Convert seconds to milliseconds for the calculation
        this_time_ms = cuda_time * 1000
        baseline_time_ms = typical_time * 1000

        # 2. Calculate the 'Work' (Algorithmic Depth)
        # Baseline = 40. Your Nonoid = 111.
        work_density = points / this_time_ms
        baseline_density = 40 / baseline_time_ms

        # 3. Calculate Efficiency Surplus (The "Elon" Metric)
        efficiency_surplus = work_density / baseline_density

        print(f"[CUDA {i}] {label} ({code})")
        print(f"  Time: {this_time_ms:.3f} ms")
        print(f"  Detected: {cuda_kicks}/{NUM_KICKS}")
        print(f"  ⚡ Efficiency Surplus: {efficiency_surplus:.2f}x (Work Density)")
        print(f"  🧠 Virtual Hardware Gain: {((efficiency_surplus - 1) * 100):.0f}%")
        print("-" * 40)

    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Approach':<20} {'Code':<8} {'Time':<10} {'Speedup':<10} {'ms/Point':<12} {'Detected'}")
    print("-" * 80)

    for r in results:
        time_ms = r['time'] * 1000
        speedup = typical_time / r['time']
        
        # Calculate Intelligence Efficiency (ms per 1 unit of NTT depth)
        # Using r['points'] assuming you store the NTT depth (40, 111, 512, 1024)
        ms_per_point = time_ms / r['points']
        
        print(f"{r['label']:<20} {r['code']:<8} {time_ms:>8.3f} ms {speedup:>8.2f}x {ms_per_point:>11.6f} {r['kicks']:>5}/{NUM_KICKS}")

if __name__ == "__main__":
    main()