%%writefile test_engine.py
import torch
import time
import sys
import glob

# 1. Setup Custom Engine
egg = glob.glob('/usr/local/lib/python3.12/dist-packages/cuboid_ops*.egg')
if egg: sys.path.append(egg[0])
import cuboid_ops

N = 512
volume = torch.ones((N, N, N), dtype=torch.int8, device="cuda")

# --- METHOD A: STANDARD PYTORCH (Direct Comparison) ---
# We use a 3D convolution to simulate searching for the 3x3x3 cuboid
def pytorch_search(vol):
    # This is how a "standard" data scientist would do it
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float16, device="cuda")
    v = vol.unsqueeze(0).unsqueeze(0).to(torch.float16)
    return torch.nn.functional.conv3d(v, kernel, stride=1)

# Warm-up both
pytorch_search(volume)
cuboid_ops.cuboid_search(volume)
torch.cuda.synchronize()

# --- TIMING ---
# Time PyTorch
start_pt = time.time()
res_pt = pytorch_search(volume)
torch.cuda.synchronize()
ms_pt = (time.time() - start_pt) * 1000

# Time Custom CUDA (Expert)
start_ex = time.time()
res_ex = cuboid_ops.cuboid_search(volume)
torch.cuda.synchronize()
ms_ex = (time.time() - start_ex) * 1000

print(f"\n[THE RESULTS]")
print(f"Standard PyTorch (GPU): {ms_pt:.4f} ms")
print(f"Expert Custom CUDA (GPU): {ms_ex:.4f} ms")
print(f"Structural Speedup: {ms_pt/ms_ex:.1f}x faster than standard GPU code")