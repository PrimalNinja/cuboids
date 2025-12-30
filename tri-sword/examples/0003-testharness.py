%%writefile testharness.py
import torch
import numpy as np
import time
import sys
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

class TriSwordBlade:
    def __init__(self, max_batch):
        self.max_batch = max_batch
        self.handle = None
    def __enter__(self):
        # Passes max_batch and default 256 threads
        self.handle = tri_sword_cuda.sharpen(self.max_batch, 256)
        return self
    def strike(self, shape, dtype, points, data):
        return tri_sword_cuda.slash(self.handle, shape, dtype, points, data)
    def __exit__(self, exc_type, exc_val, exc_tb):
        tri_sword_cuda.sheath(self.handle)

def generate_audio_spectrograms(batch_size=1000000):
    spectrograms = torch.rand(batch_size, 256, device='cuda') * 0.5
    kick_indices = np.random.choice(batch_size, 500, replace=False)
    for idx in kick_indices:
        spectrograms[idx, :20] = torch.rand(20, device='cuda') * 3.0 + 2.0
    return spectrograms, set(kick_indices)

def main():
    BATCH_SIZE = 1000000
    NUM_KICKS = 500
    
    data, _ = generate_audio_spectrograms(BATCH_SIZE)
    data = data.contiguous()

    float_data = data.clone().float().cuda()
    binary_data = (data * 20.0).to(torch.uint8).contiguous()
    ternary_data = (data * 20.0).to(torch.int8).contiguous()

    results = []

    with TriSwordBlade(max_batch=BATCH_SIZE) as blade:
        print("\nWarming up GPU...")
        _ = blade.strike('V', 'F', 512, float_data)
        torch.cuda.synchronize()

        # 1. TYPICAL BASELINE (Primary Reference)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        typical_res = blade.strike('V', 'F', 512, float_data)
        end.record()
        torch.cuda.synchronize()
        typical_time = start.elapsed_time(end) / 1000.0
        
        results.append({
            'label': 'Typical (Baseline)', 'code': 'TYP512', 
            'time': typical_time, 'kicks': typical_res[0].item(), 'points': 512
        })
        print(f"[Typical] Time: {typical_time*1000:.3f} ms\n")

        # 2. FULL SWEEP RESTORED
        configs = [
            ('V', 'B', 40, 'VBT40'),   ('V', 'T', 40, 'VTT40'),
            ('T', 'B', 40, 'TBT40'),   ('T', 'T', 40, 'TTT40'),
            ('P', 'B', 40, 'PBT40'),   ('P', 'T', 40, 'PTT40'),
            ('C', 'B', 80, 'CBT80'),   ('C', 'T', 80, 'CTT80'),
            ('N', 'B', 111, 'NBT111'), ('N', 'T', 111, 'NTT111'),
            ('S', 'B', 111, 'SCY111'),
            ('V', 'B', 512, 'VBT512'), ('V', 'T', 512, 'VTT512'),
            ('T', 'B', 512, 'TBT512'), ('T', 'T', 512, 'TTT512'),
            ('P', 'B', 512, 'PBT512'), ('P', 'T', 512, 'PTT512'),
            ('C', 'B', 512, 'CBT512'), ('C', 'T', 512, 'CTT512'),
            ('N', 'B', 512, 'NBT512'), ('N', 'T', 512, 'NTT512'),
            ('N', 'T', 1024, 'NTT1024'), ('N', 'T', 4096, 'NTT4096'), 
            ('N', 'T', 16384, 'NTT16384'), ('N', 'T', 32768, 'NTT32768')
        ]

        for i, (shape, dtype, points, label) in enumerate(configs, 1):
            input_tensor = binary_data if dtype == 'B' else ternary_data
            torch.cuda.synchronize()
            start.record()
            res = blade.strike(shape, dtype, points, input_tensor)
            end.record()
            torch.cuda.synchronize()
            
            elapsed = start.elapsed_time(end) / 1000.0
            results.append({
                'label': label, 'code': f"{shape}{dtype}{points}",
                'time': elapsed, 'kicks': res[0].item(), 'points': points
            })
            
            # Density metrics
            work_density = points / (elapsed * 1000)
            baseline_density = 512 / (typical_time * 1000)
            eff = work_density / baseline_density
            gain = (eff - 1) * 100
            
            print(f"[CUDA {i}] {label} | {elapsed*1000:.3f} ms | Eff: {eff:.2f}x | Gain: {gain:.0f}%")
            sys.stdout.flush()

    # --- SUMMARY TABLE ---
    print(f"\n{'='*95}")
    print(f"{'Approach':<20} {'Code':<10} {'Time':<12} {'Speedup':<10} {'ms/Point':<12} {'Detected'}")
    print("-" * 95)
    
    baseline_t = results[0]['time']
    for r in results:
        t_ms = r['time'] * 1000
        speedup = baseline_t / r['time']
        mpp = t_ms / r['points']
        print(f"{r['label']:<20} {r['code']:<10} {t_ms:>8.3f} ms {speedup:>8.2f}x {mpp:>11.6f} {int(r['kicks']):>8}")

if __name__ == "__main__":
    main()