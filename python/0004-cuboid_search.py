%%writefile test_engine.py
import torch
import time
import sys
import glob

# 1. Link your Expert Engine
egg = glob.glob('/usr/local/lib/python3.12/dist-packages/cuboid_ops*.egg')
if egg: sys.path.append(egg[0])
import cuboid_ops

N = 512
volume = torch.ones((N, N, N), dtype=torch.int8, device="cuda")

# --- METHOD A: PyTorch with Structural Skipping (Stride=3) ---
def pytorch_structural_search(vol):
    with torch.no_grad():
        v = vol.unsqueeze(0).unsqueeze(0).float()
        kernel = torch.ones((1, 1, 3, 3, 3), device="cuda").float()
        # We add STRIDE=3 to match your Cube81 logic
        return torch.nn.functional.conv3d(v, kernel, stride=3)

# Warm-up
pytorch_structural_search(volume)
cuboid_ops.cuboid_search(volume)
torch.cuda.synchronize()

# --- TIMING ---
# Time PyTorch (Now with Stride=3)
s1 = time.time()
pytorch_structural_search(volume)
torch.cuda.synchronize()
ms_pt = (time.time() - s1) * 1000

# Time Custom CUDA (Your Cube81 Logic)
s2 = time.time()
cuboid_ops.cuboid_search(volume)
torch.cuda.synchronize()
ms_ex = (time.time() - s2) * 1000

print(f"\n" + "="*40)
print(f"PyTorch Structural (Stride 3): {ms_pt:8.4f} ms")
print(f"Expert Custom (Cube81):       {ms_ex:8.4f} ms")
print(f"The 'Expert' Multiplier:      {ms_pt/ms_ex:8.1f}x")
print("="*40)