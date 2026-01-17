%%writefile zoscii_testharness.py
import torch
import numpy as np
import sys
import os
from torch.utils.cpp_extension import load

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("JIT compiling ZOSCII Tri-Sword Engine...")
zoscii_cuda = load(
    name='zoscii_cuda',
    sources=['zoscii_trisword.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)

class ZOSCIICore:
    def __init__(self, rom_file="image.jpg", max_batch=1000000):
        rom_size = 65536
        if not os.path.exists(rom_file):
            print(f"Creating test ROM: {rom_file}")
            with open(rom_file, "wb") as f:
                f.write(np.random.randint(0, 256, rom_size, dtype=np.uint8).tobytes())
        
        with open(rom_file, 'rb') as f:
            rom_bytes = f.read(rom_size)
        
        if len(rom_bytes) < rom_size:
            padding = np.random.randint(0, 256, rom_size - len(rom_bytes), dtype=np.uint8)
            rom_bytes += padding.tobytes()
        
        rom_np = np.frombuffer(rom_bytes[:rom_size], dtype=np.uint8).astype(np.int8)
        
        self.handle = zoscii_cuda.sharpen(max_batch, 
                                         torch.from_numpy(rom_np).contiguous(),
                                         rom_size)
        self.rom_np = rom_np
        print(f"ZOSCII Tri-Sword initialized with {max_batch:,} batch capacity")
    
    def encode(self, messages):
        return zoscii_cuda.slash_encode(self.handle, messages)
    
    def decode(self, encoded):
        return zoscii_cuda.slash_decode(self.handle, encoded)
    
    def __del__(self):
        zoscii_cuda.sheath(self.handle)

def run_zoscii_benchmark(batch_size=10000, msg_len=512):
    # FIX: Using signed range (-128, 128) to satisfy torch.int8 bounds
    test_data = torch.randint(-128, 128, (batch_size, msg_len), 
                              dtype=torch.int8, device='cuda')
    
    core = ZOSCIICore("image.jpg", batch_size)
    
    # Warmup
    _ = core.encode(test_data[:100])
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    encoded = core.encode(test_data)
    end.record()
    torch.cuda.synchronize()
    encode_time = start.elapsed_time(end)
    
    start.record()
    decoded = core.decode(encoded)
    end.record()
    torch.cuda.synchronize()
    decode_time = start.elapsed_time(end)
    
    correct = torch.all(test_data == decoded)
    total_bytes = batch_size * msg_len
    encode_mbps = total_bytes / (encode_time / 1000) / 1e6
    decode_mbps = total_bytes / (decode_time / 1000) / 1e6
    
    # Baseline for JS comparison
    js_encode_mbps = 0.211864 
    js_decode_mbps = 0.724637 
    
    return {
        'batch_size': batch_size,
        'msg_len': msg_len,
        'encode_time_ms': encode_time,
        'decode_time_ms': decode_time,
        'encode_mbps': encode_mbps,
        'decode_mbps': decode_mbps,
        'js_encode_x': encode_mbps / js_encode_mbps,
        'js_decode_x': decode_mbps / js_decode_mbps,
        'correct': correct
    }

def main():
    print(f"\n{'='*90}")
    print(f"ZOSCII 64KB TRI-SWORD PERFORMANCE AUDIT")
    print(f"{'='*90}")
    
    print(f"{'Batch':<10} | {'Encode (ms)':<12} | {'Decode (ms)':<12} | "
          f"{'Encode MB/s':<12} | {'Decode MB/s':<12} | {'JS Encode x':<12} | {'Correct'}")
    print("-" * 110)
    
    batch_sizes = [5000, 50000, 500000, 2000000, 3500000]
    
    for batch in batch_sizes:
        try:
            results = run_zoscii_benchmark(batch, 512)
            print(f"{batch:<10} | {results['encode_time_ms']:>10.2f} | {results['decode_time_ms']:>10.2f} | "
                  f"{results['encode_mbps']:>10.1f} | {results['decode_mbps']:>10.1f} | "
                  f"{results['js_encode_x']:>10.0f}x | {'✓' if results['correct'] else '✗'}")
        except Exception as e:
            print(f"{batch:<10} | ERROR: {str(e)}")
        torch.cuda.empty_cache()

    print(f"\nRandomness Verification:")
    test_core = ZOSCIICore("image.jpg", 1000)
    # FIX: Signed range check here as well
    test_msg = torch.randint(-128, 128, (1, 512), dtype=torch.int8, device='cuda')
    
    e1 = test_core.encode(test_msg)
    e2 = test_core.encode(test_msg)
    same = torch.sum(e1 == e2).item()
    print(f"  Same positions: {same}/512 (Ideal: <5)")
    print(f"  True ZOSCII: {'✓' if same < 10 else '✗'}")

if __name__ == "__main__":
    main()