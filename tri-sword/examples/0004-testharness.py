%%writefile testharness.py
import torch
import numpy as np
import sys
from torch.utils.cpp_extension import load

print("JIT compiling Sovereign DNA Engine...")
tri_sword_cuda = load(
    name='tri_sword_cuda',
    sources=['tri_sword.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class TriSwordBlade:
    def __init__(self, max_batch):
        self.max_batch = max_batch
        self.handle = tri_sword_cuda.sharpen(self.max_batch, 256)
    def strike(self, points, data):
        return tri_sword_cuda.slash(self.handle, 'D', 'B', points, data)
    def __del__(self):
        tri_sword_cuda.sheath(self.handle)

def main():
    BATCH_SIZE = 1000000 
    dna_universe = (torch.rand(BATCH_SIZE, 256, device='cuda') > 0.8).to(torch.uint8).contiguous()
    blade = TriSwordBlade(BATCH_SIZE)

    print(f"\n--- DNA PERSISTENCE AUDIT (Batch: {BATCH_SIZE}) ---")
    
    # Baseline: 1 Generation
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); blade.strike(1, dna_universe); end.record()
    torch.cuda.synchronize()
    base_t = start.elapsed_time(end)

    results = []
    for gens in [1, 10, 100, 500, 1000, 10000]:
        test_grid = dna_universe.clone()
        start.record(); blade.strike(gens, test_grid); end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        
        mpp = t / gens # ms per generation
        persistence = (base_t) / mpp
        results.append((gens, t, mpp, persistence))
        print(f"Gens: {gens:4d} | Total: {t:8.2f}ms | ms/Gen: {mpp:8.4f} | Gain: {persistence:6.2f}x")

    print(f"\n{'='*75}\nFinal Audit: Maximum Silicon Persistence Achieved: {results[-1][3]:.2f}x\n{'='*75}")

if __name__ == "__main__":
    main()