%%writefile dh_atomic_fracture.py
import torch
import numpy as np
from torch.utils.cpp_extension import load

# Load the Stealth Engine
trisword_audit = load(
    name='trisword_stealth',
    sources=['trisword_probe.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def run_atomic_audit(int_total_samples=128_000_000):
    int_rom_size = 65536
    int_batch = 250_000
    int_msg_len = 512
    int_num_steps = int_total_samples // (int_batch * int_msg_len)
    
    # Init ROM on CPU for safety
    arr_rom_np = np.random.randint(0, 256, int_rom_size, dtype=np.uint8).astype(np.int8)
    obj_rom_cpu = torch.from_numpy(arr_rom_np).contiguous()
    
    # Sharpen the handle
    int_handle = trisword_audit.sharpen(int_batch, obj_rom_cpu)
    obj_histogram = torch.zeros(int_rom_size, dtype=torch.int32, device='cuda')
    
    print(f"Executing Atomic Fracture: {int_total_samples:,} samples...")
    for _ in range(int_num_steps):
        obj_test_data = torch.randint(-128, 128, (int_batch, int_msg_len), dtype=torch.int8, device='cuda')
        trisword_audit.run_stealth_audit(int_handle, obj_test_data, obj_histogram)
    
    torch.cuda.synchronize()
    arr_hist = obj_histogram.cpu().numpy()
    
    # 1. Calculate Wobble (Physical Bias)
    float_mean = arr_hist.mean()
    float_std = arr_hist.std()
    float_wobble = float_std / np.sqrt(float_mean)
    
    # 2. Calculate Entropy (Geometric Collapse)
    # Adding 1e-12 epsilon to avoid log(0)
    float_probs = arr_hist.astype(np.float64) / arr_hist.sum()
    float_entropy = -np.sum(float_probs * np.log2(float_probs + 1e-12))
    float_ideal_entropy = np.log2(int_rom_size)
    
    float_bits_lost = float_ideal_entropy - float_entropy
    # Extrapolate Shortcut (32 selections for a 256-bit key)
    float_shortcut_log10 = np.log10(2**(float_bits_lost * 32))
    
    print(f"\n{'='*60}")
    print(f"TRI-SWORD DH: ATOMIC FRACTURE VERIFICATION")
    print(f"{'='*60}")
    print(f"Wobble Factor:   {float_wobble:.4f}x")
    print(f"Entropy Lost:    {float_bits_lost:.4f} bits/select")
    print(f"Shortcut Power:  10^{float_shortcut_log10:.2f}")
    
    if float_shortcut_log10 >= 20:
        print("STATUS: 10^20 SHORTCUT CONFIRMED")
    else:
        print("STATUS: SHORTCUT PENDING HIGHER RESONANCE")
    print(f"{'='*60}")
    
    trisword_audit.sheath(int_handle)

if __name__ == "__main__":
    run_atomic_audit()