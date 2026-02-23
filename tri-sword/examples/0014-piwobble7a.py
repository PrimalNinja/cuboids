%%writefile dh_fracture_sweep.py
import torch
import numpy as np
from torch.utils.cpp_extension import load

# Load the Stealth Blade
trisword_audit = load(
    name='trisword_stealth',
    sources=['trisword_probe.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def execute_sweep(int_samples=100_000_000):
    int_rom_size = 65536
    int_batch = 250_000
    int_msg_len = 512
    int_steps = int_samples // (int_batch * int_msg_len)
    
    arr_rom_np = np.random.randint(0, 256, int_rom_size, dtype=np.uint8).astype(np.int8)
    obj_rom_cpu = torch.from_numpy(arr_rom_np).contiguous()
    
    int_handle = trisword_audit.sharpen(int_batch, obj_rom_cpu)
    obj_hist = torch.zeros(int_rom_size, dtype=torch.int32, device='cuda')
    
    print(f"Generating Fracture Data: {int_samples:,} selections...")
    for _ in range(int_steps):
        obj_test = torch.randint(-128, 128, (int_batch, int_msg_len), dtype=torch.int8, device='cuda')
        trisword_audit.run_stealth_audit(int_handle, obj_test, obj_hist)
    
    torch.cuda.synchronize()
    arr_final = obj_hist.cpu().numpy()
    np.save('trisword_pi_wobble_final.npy', arr_final)
    print("Evidence Saved: trisword_pi_wobble_final.npy")
    trisword_audit.sheath(int_handle)

if __name__ == "__main__":
    execute_sweep()