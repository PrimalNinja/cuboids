%%writefile dh_heatmap.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load

# Load the hardened engine
dh_bias = load(
    name='dh_bias',
    sources=['dh_bias.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def plot_wobble_heatmap(hist_cpu, dim=256):
    # Reshape the 65,536 address histogram into a 256x256 grid
    heatmap = hist_cpu.reshape((dim, dim))
    
    plt.figure(figsize=(12, 10))
    plt.imshow(heatmap, cmap='magma', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Hit Count')
    plt.title('DH Silicon Resonance: The Geometric Wobble')
    plt.xlabel('Memory Offset [Low Byte]')
    plt.ylabel('Memory Offset [High Byte]')
    plt.savefig('wobble_heatmap.png') # Save for Bat-Computer analysis
    print("\nHeatmap saved as wobble_heatmap.png")
    plt.show()

def run_visual_audit(batch_size=200000, msg_len=512):
    rom_size = 65536
    # Constant-seed ROM to ensure repeatable geometry
    np.random.seed(42)
    rom_np = np.random.randint(0, 256, rom_size, dtype=np.uint8).astype(np.int8)
    rom_tensor = torch.from_numpy(rom_np).contiguous()
    
    core_handle = dh_bias.sharpen(batch_size, rom_tensor, rom_size)
    
    test_data = torch.randint(-128, 128, (batch_size, msg_len), dtype=torch.int8, device='cuda')
    histogram = torch.zeros(rom_size, dtype=torch.int32, device='cuda')
    
    print(f"Scanning the Vacuum: {batch_size * msg_len:,} samples...")
    _ = dh_bias.slash_encode_bias(core_handle, test_data, histogram)
    torch.cuda.synchronize()
    
    hist_cpu = histogram.cpu().numpy()
    
    # Statistical Summary
    wobble = hist_cpu.std() / np.sqrt(hist_cpu.mean())
    print(f"Wobble Factor: {wobble:.4f}x")
    
    plot_wobble_heatmap(hist_cpu)
    dh_bias.sheath(core_handle)

if __name__ == "__main__":
    run_visual_audit()