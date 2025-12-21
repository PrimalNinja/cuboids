%%writefile test_engine.py
from torch.utils.cpp_extension import load
import torch
import time

# Load the expert kernel
expert = load(name="expert", sources=["engine.cu"], verbose=False)

# Setup: 10M voxels
N = 10_000_000
data = torch.zeros(N, dtype=torch.uint8, device='cuda')
data[500:502] = 1  # The pattern to tag

# Warmup
for _ in range(3):
    expert.tag(data)
    res_pt = data.clone()
    res_pt[:-1] = torch.where((data[:-1] == 1) & (data[1:] == 1), 2, data[:-1])

torch.cuda.synchronize()
time.sleep(1)  # let GPU cool a bit

# Multi-run benchmark
num_runs = 10
expert_times = []
pytorch_times = []

for _ in range(num_runs):
    # Expert
    start = time.time()
    res_expert = expert.tag(data)
    torch.cuda.synchronize()
    expert_times.append((time.time() - start) * 1000)

    # PyTorch typical
    start = time.time()
    res_pt = data.clone()
    res_pt[:-1] = torch.where((data[:-1] == 1) & (data[1:] == 1), 2, data[:-1])
    torch.cuda.synchronize()
    pytorch_times.append((time.time() - start) * 1000)

# Verify correctness
expert_tagged = (res_expert == 2).sum().item()
pt_tagged = (res_pt == 2).sum().item()
assert expert_tagged == pt_tagged == 1

print("--- PROOF OF LIFE: TAGGING EDITION ---")
print(f"Patterns Tagged: {expert_tagged}")
print(f"Expert Latency : {sum(expert_times)/num_runs:.4f} ms (avg over {num_runs} runs)")
print(f"PyTorch Latency: {sum(pytorch_times)/num_runs:.4f} ms (avg over {num_runs} runs)")
print(f"Expert Speedup : {sum(pytorch_times)/sum(expert_times):.1f}x")