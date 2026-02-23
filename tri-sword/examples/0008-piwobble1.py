%%writefile trisword_audit.py
import torch
import numpy as np
import os
from torch.utils.cpp_extension import load

print("Compiling Tri-Sword DH Truncation Audit Blade...")
trisword = load(
    name='trisword_cuda',
    sources=['trisword_dh.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def run_geometric_audit(batch_size=1000000, msg_len=512):
    rom_size = 65536
    # Constant ROM to map the static vertices of the polygon
    np.random.seed(42)
    rom_np = np.random.randint(0, 256, rom_size, dtype=np.uint8).astype(np.int8)
    
    handle = trisword.sharpen(batch_size, torch.from_numpy(rom_np).contiguous(), rom_size)
    
    # Audit Data: Simulating DH coordinate selections
    test_msg = torch.randint(-128, 128, (batch_size, msg_len), dtype=torch.int8, device='cuda')
    
    print(f"Executing PI-Truncation Scan: {batch_size * msg_len:,} samples...")
    encoded = trisword.ts_encode(handle, test_msg)
    torch.cuda.synchronize()
    
    # Calculate Geometric Bias (The Wobble)
    flat_enc = encoded.view(-1).cpu().numpy()
    counts = np.bincount(flat_enc, minlength=rom_size)
    
    mean_hits = counts.mean()
    std_dev = counts.std()
    wobble = std_dev / np.sqrt(mean_hits)
    
    print(f"\n{'='*60}")
    print(f"TRI-SWORD DH GEOMETRIC COLLAPSE RESULTS")
    print(f"{'='*60}")
    print(f"Truncation Wobble: {wobble:.4f}x")
    print(f"Max Resonance Peak: {np.max(counts)} hits")
    print(f"Primary Vertex: Addr {np.argmax(counts)}")
    
    trisword.sheath(handle)

if __name__ == "__main__":
    run_geometric_audit()