%%writefile dh_visual_audit.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load

# Load the existing extension
dh_bias = load(
    name='dh_bias',
    sources=['dh_bias.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def run_visual_audit(batch_size=200000, msg_len=512):
    rom_size = 65536
    # Constant ROM to see the "Fixed Geometry"
    np.random.seed(42)
    rom_np = np.random.randint(0, 256, rom_size, dtype=np.uint8).astype(np.int8)
    rom_tensor = torch.from_numpy(rom_np).contiguous()
    
    core_handle = dh_bias.sharpen(batch_size, rom_tensor, rom_size)
    
    test_data = torch.randint(-128, 128, (batch_size, msg_len), dtype=torch.int8, device='cuda')
    histogram = torch.zeros(rom_size, dtype=torch.int32, device='cuda')
    
    print(f"Mapping the Polygon on T4...")
    _ = dh_bias.slash_encode_bias(core_handle, test_data, histogram)
    torch.cuda.synchronize()
    
    hist_cpu = histogram.cpu().numpy()
    
    # 1. Plot Heatmap
    heatmap = hist_cpu.reshape((256, 256))
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='hot', aspect='auto')
    plt.colorbar(label='Hit Count')
    plt.title(f'DH Silicon Resonance (Wobble: {hist_cpu.std()/np.sqrt(hist_cpu.mean()):.2f}x)')
    plt.savefig('wobble_map.png')
    print("Heatmap saved: wobble_map.png")

    # 2. Identify Hotspots
    top_indices = np.argsort(hist_cpu)[-10:][::-1]
    print("\nTOP 10 RESONANCE PEAKS (ROM ADDRESSES):")
    for i in top_indices:
        print(f"Addr: {i} | Hits: {hist_cpu[i]} (Mean: {hist_cpu.mean():.1f})")

    dh_bias.sheath(core_handle)

if __name__ == "__main__":
    run_visual_audit()