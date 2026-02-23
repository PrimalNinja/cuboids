%%writefile dh_bias_test.py
import torch
import numpy as np
import os
from torch.utils.cpp_extension import load

# Initialize the JIT Build
dh_bias = load(
    name='dh_bias',
    sources=['dh_bias.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def run_bias_audit(batch_size=100000, msg_len=512):
    rom_size = 65536
    # Creating the ROM on CPU first to satisfy the sharpen signature
    rom_np = np.random.randint(0, 256, rom_size, dtype=np.uint8).astype(np.int8)
    rom_tensor = torch.from_numpy(rom_np).contiguous()
    
    # Secure the handle
    core_handle = dh_bias.sharpen(batch_size, rom_tensor, rom_size)
    
    # 51.2 Million samples is enough for the first Wobble check
    test_data = torch.randint(-128, 128, (batch_size, msg_len), dtype=torch.int8, device='cuda')
    histogram = torch.zeros(rom_size, dtype=torch.int32, device='cuda')
    
    print(f"Executing Bias Audit on Tesla T4...")
    try:
        _ = dh_bias.slash_encode_bias(core_handle, test_data, histogram)
        torch.cuda.synchronize()
        
        hist_cpu = histogram.cpu().numpy()
        mean_hits = hist_cpu.mean()
        std_dev = hist_cpu.std()
        expected_std = np.sqrt(mean_hits) 
        wobble_factor = std_dev / expected_std
        
        print(f"\n{'='*60}")
        print(f"DH WOBBLE AUDIT RESULTS")
        print(f"{'='*60}")
        print(f"Mean Hits per Addr: {mean_hits:.2f}")
        print(f"Observed Std Dev:   {std_dev:.2f}")
        print(f"Ideal Std Dev:      {expected_std:.2f}")
        print(f"Wobble Factor:      {wobble_factor:.4f}x")
        
    finally:
        dh_bias.sheath(core_handle)

if __name__ == "__main__":
    run_bias_audit()