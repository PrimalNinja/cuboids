%%writefile test_engine.py
from torch.utils.cpp_extension import load
import torch
import time

# IGNITION: Register the jumpblock we just wrote
expert = load(name="expert", sources=["engine.cu"])

# FRAMEBUFFER: 10M voxels on the GPU
data = torch.zeros(10_000_000, dtype=torch.uint8, device='cuda')
data[500:502] = 1 # Seed the "Life" pattern

# EXECUTION
start = time.time()
res = expert.tag(data)
torch.cuda.synchronize()
elapsed = (time.time() - start) * 1000

print(f"--- PROOF OF LIFE ---")
print(f"Patterns Successfully Tagged: {(res == 2).sum().item()}")
print(f"Hardware Latency:            {elapsed:.4f}ms")