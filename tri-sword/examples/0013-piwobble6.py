%%writefile dh_save_hist.py
import torch
import numpy as np
from torch.utils.cpp_extension import load

dh_bias = load(name='dh_bias', sources=['dh_bias.cu'])

def save_audit_data(batch_size=200000, msg_len=512):
    rom_size = 65536
    np.random.seed(42)
    rom_np = np.random.randint(0, 256, rom_size, dtype=np.uint8).astype(np.int8)
    core_handle = dh_bias.sharpen(batch_size, torch.from_numpy(rom_np), rom_size)
    
    test_data = torch.randint(-128, 128, (batch_size, msg_len), dtype=torch.int8, device='cuda')
    histogram = torch.zeros(rom_size, dtype=torch.int32, device='cuda')
    
    _ = dh_bias.slash_encode_bias(core_handle, test_data, histogram)
    torch.cuda.synchronize()
    
    # Save the raw distribution to disk
    np.save('hist_data.npy', histogram.cpu().numpy())
    print("Forensic data locked: hist_data.npy")
    dh_bias.sheath(core_handle)

if __name__ == "__main__":
    save_audit_data()