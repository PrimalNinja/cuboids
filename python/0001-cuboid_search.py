%%writefile test_engine.py
import sys
import torch
import time
import glob

# Find the engine we installed via the shell
egg = glob.glob('/usr/local/lib/python3.12/dist-packages/cuboid_ops*.egg')
if egg: sys.path.append(egg[0])

import cuboid_ops

N = 512
volume = torch.ones((N, N, N), dtype=torch.int8, device="cuda")

# Warm-up
cuboid_ops.cuboid_search(volume)

# Benchmark
torch.cuda.synchronize()
start = time.time()
res = cuboid_ops.cuboid_search(volume)
torch.cuda.synchronize()
ms = (time.time() - start) * 1000

print(f"\n[RESULTS]")
print(f"Time: {ms:.4f} ms")
print(f"Speedup: {1600/ms:.1f}x vs CPU")