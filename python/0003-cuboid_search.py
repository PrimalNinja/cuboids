%%writefile test_engine.py
import torch
import time
import sys
import glob

# 1. Link the expert engine
egg = glob.glob('/usr/local/lib/python3.12/dist-packages/cuboid_ops*.egg')
if egg: sys.path.append(egg[0])
import cuboid_ops

N = 512
volume = torch.ones((N, N, N), dtype=torch.int8, device="cuda")

# --- METHOD A: Standard PyTorch (The 'Standard' Expert) ---
# We use a 3D convolution to simulate the 3x3x3 search
def pytorch_search(vol):
    with torch.no_grad():
        v = vol.unsqueeze(0).unsqueeze(0).float()
        kernel = torch.ones((1, 1, 3, 3, 3), device="cuda").float()
        return torch.nn.functional.conv3d(v, kernel, stride=1)

# Warm-up (Load kernels into GPU)
pytorch_search(volume)
cuboid_ops.cuboid_search(volume)
torch.cuda.synchronize()

# --- TIMING ---
# Time PyTorch
s1 = time.time()
res_pt = pytorch_search(volume)
torch.cuda.synchronize()
ms_pt = (time.time() - s1) * 1000

# Time Custom CUDA (Your Structural Logic)
s2 = time.time()
res_ex = cuboid_ops.cuboid_search(volume)
torch.cuda.synchronize()
ms_ex = (time.time() - s2) * 1000

print(f"\n" + "="*35)
print(f"Standard PyTorch (GPU): {ms_pt:8.4f} ms")
print(f"Expert Custom (GPU):    {ms_ex:8.4f} ms")
print(f"The 'Structural' Gap:   {ms_pt/ms_ex:8.1f}x faster")
print("="*35)