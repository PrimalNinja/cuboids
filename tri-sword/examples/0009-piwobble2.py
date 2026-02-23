%%writefile dh_precision_audit.py
import torch
import numpy as np
from torch.utils.cpp_extension import load

# JIT Compile the Stealth Blade
trisword_audit = load(
    name='trisword_stealth',
    sources=['trisword_probe.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def run_dh_precision_audit(total_samples=512_000_000, batch_size=250_000):
    int_rom_size = 65536
    int_msg_len = 512
    int_num_batches = total_samples // (batch_size * int_msg_len)
    
    # Initialize ROM on CPU to prevent the initialization Segfault
    arr_rom_np = np.random.randint(0, 256, int_rom_size, dtype=np.uint8).astype(np.int8)
    obj_rom_cpu = torch.from_numpy(arr_rom_np).contiguous()
    
    int_handle = trisword_audit.sharpen(batch_size, obj_rom_cpu)
    obj_histogram = torch.zeros(int_rom_size, dtype=torch.int32, device='cuda')
    
    print(f"Executing Stealth Audit: {total_samples:,} samples total...")
    
    for i in range(int_num_batches):
        obj_test_data = torch.randint(-128, 128, (batch_size, int_msg_len), dtype=torch.int8, device='cuda')
        trisword_audit.run_stealth_audit(int_handle, obj_test_data, obj_histogram)
        
        if i % 10 == 0:
            torch.cuda.synchronize()
            float_progress = ((i + 1) / int_num_batches) * 100
            print(f"  Progress: {float_progress:.1f}%")
            
    torch.cuda.synchronize()
    arr_hist_cpu = obj_histogram.cpu().numpy()
    
    float_mean_hits = arr_hist_cpu.mean()
    float_std_dev = arr_hist_cpu.std()
    float_wobble = float_std_dev / np.sqrt(float_mean_hits)
    
    print(f"\n{'='*60}")
    print(f"TRI-SWORD DH GEOMETRIC COLLAPSE: PI TRUNCATION AUDIT")
    print(f"{'='*60}")
    print(f"Mean Hits:      {float_mean_hits:.2f}")
    print(f"Wobble Factor:  {float_wobble:.4f}x")
    print(f"Primary Vertex: Addr {np.argmax(arr_hist_cpu)}")
    print(f"{'='*60}")
    
    trisword_audit.sheath(int_handle)

if __name__ == "__main__":
    run_dh_precision_audit()