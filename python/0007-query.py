%%writefile test_engine.py
import torch
from torch.utils.cpp_extension import load
import time

# JIT Compile the expert engine
expert_engine = load(name="expert_engine", sources=["engine.cu"], verbose=True)

# 1. Setup Data (10 million voxels)
data = torch.zeros(10_000_000, dtype=torch.uint8, device='cuda')
data[500:502] = 1 # The "Life" we are looking for

# 2. Traditional PyTorch Method (The "Abstraction Layer")
# Even a vectorized check has overhead because it creates a new boolean tensor
start = time.time()
found_trad = ((data[:-1] > 0) & (data[1:] > 0)).sum(dtype=torch.int32)
trad_time = (time.time() - start) * 1000

# 3. Expert Method (The "Tunnel-Vision" API)
start = time.time()
found_expert = expert_engine.query(data)
expert_time = (time.time() - start) * 1000

print(f"--- PROOF OF LIFE ---")
print(f"Traditional Found: {found_trad} | Time: {trad_time:.4f}ms")
print(f"Expert Found:      {found_expert} | Time: {expert_time:.4f}ms")
print(f"Expert Speedup:    {trad_time / expert_time:.1f}x")