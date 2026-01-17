%%writefile upload.py
#!/usr/bin/env python3
"""
ZOSCII 64KB ROM Setup for Google Colab
Run this in a Colab CELL, not in terminal!
"""

import numpy as np
import os

def create_test_rom(filename="image.jpg"):
    """Create a synthetic 64KB test ROM with good byte distribution"""
    print(f"Creating synthetic 64KB ROM: {filename}")
    
    # Create random 64KB
    rom_data = np.random.randint(0, 256, 65536, dtype=np.uint8)
    
    # Ensure every byte appears at least ~200 times (close to uniform)
    for byte_val in range(256):
        count = np.sum(rom_data == byte_val)
        if count < 200:
            # Replace some random positions with missing byte
            indices = np.random.choice(65536, 220 - count, replace=False)
            rom_data[indices] = byte_val
    
    # Save to file
    with open(filename, "wb") as f:
        f.write(rom_data.tobytes())
    
    # Analyze distribution
    counts = np.bincount(rom_data, minlength=256)
    print(f"✓ Created {filename}")
    print(f"  Size: {os.path.getsize(filename):,} bytes")
    print(f"  Bytes 0-255 appear: {counts.min()} to {counts.max()} times")
    print(f"  Average: {counts.mean():.1f} (ideal: 256)")
    print(f"  Std deviation: {counts.std():.1f}")
    
    return True

def main():
    """Main setup - call this from a Colab cell"""
    print("=" * 60)
    print("ZOSCII 64KB ROM Setup")
    print("=" * 60)
    
    # Always create test ROM for simplicity
    create_test_rom("image.jpg")
    
    print("\nTo upload your own file in Colab:")
    print("1. Use the 📁 folder icon on left")
    print("2. Drag & drop your file")
    print("3. Rename it to 'image.jpg' or update the path in test script")
    
    return "image.jpg"

if __name__ == "__main__":
    # This will only work properly when imported in Colab
    # For terminal use, just create the test ROM
    create_test_rom("image.jpg")