# Cuboids: A GPU-Accelerated Ternary Spatial Computing Framework

## 🧠 The Paradigm Shift
**Stop moving data. Start evolving perception.**

Cuboids demonstrates a fundamental shift in 3D spatial computing. Traditional methods physically transform and write massive 3D datasets (voxels) to memory for each new viewpoint. Cuboids redefines the problem: it evolves compact **"Spatial DNA"** parameters—a mere 6 numbers defining a 3D view—while the data stays put. This transforms the GPU from a number cruncher into an **autonomous spatial reasoning co-processor**.

## ⚡️ Validated Performance at Scale
**The core innovation has been tested at production-relevant scale.** The latest benchmarks (N=512, 134 million voxels) show the DNA paradigm achieving a **consistent 18-20x speedup** over an optimized traditional baseline.

| Metric | Traditional (Transform & Write) | Cuboids (Evolve Perception) | Advantage |
| :--- | :--- | :--- | :--- |
| **Workload** | 100 rotation+score cycles on 134M voxels | 100 virtual evaluation cycles | Identical problem |
| **Time** | ~19,514 ms | **~1,030 ms** | **18.94x faster** |
| **Key Bottleneck** | Memory Bandwidth (~13.4 GB of writes) | Compute / ALU Throughput | Operates in a faster hardware regime |
| **Kernel Launches** | 200 | **1** | Massive overhead eliminated |

**Why it scales:** The performance gap grows with problem size. As data volume increases, the traditional approach hits a memory bandwidth wall, while Cuboids' cost scales with compute—a fundamental advantage on modern hardware.

## ⚙️ How It Works: The Technical Core
1.  **Ternary Logic Substrate**: Voxels are `int8_t` with values `-1, 0, 1` (Inhibit/Empty/Excite). This is a dense, efficient representation for spatial correlation, offering a 4x memory saving over `float32`.
2.  **Spatial DNA**: A 3D perspective is encoded in just 6 evolvable parameters (`tx, ty, tz, rx, ry, rz`). The algorithm searches for the DNA that yields the optimal "view" of the data.
3.  **Persistent GPU-Resident Evolution**: The entire evolutionary search—hundreds of generations—runs in persistent GPU kernels. This eliminates costly CPU-GPU synchronization and kernel launch overhead.
4.  **Spatial Instruction Set**: Kernels implement instructions like `ROTX`, `MADD`, and `SCORE`, forming a Turing-complete instruction set for volumetric reasoning that operates directly on the ternary substrate.

## 🛠️ Project State: A Proven Engine, An Open Challenge
This repository contains **79 progressively optimized CUDA files** documenting the complete journey from a JavaScript concept to validated hardware performance. It is a **proven prototype and an open benchmarking challenge**.

-   **✅ What's Validated**: The **DNA paradigm works and is fast**. Files 0001-0077 are fully tested. The architectural advantage is clear and scales to 134M voxels.
-   **🎯 The Open Challenge**: We invite the community to determine the **absolute performance ceiling**. Both the Cuboids ("DNA") code and the "traditional" baseline have optimization headroom. Can expert CUDA optimization close the gap, or does the paradigm's structural advantage hold?
    -   **Optimize the traditional baseline** with expert techniques.
    -   **Push the DNA implementation** further.
    -   **Run the definitive fair race** on A100, H100, and other architectures.

## 🛠 Getting Started
**Prerequisites**: A CUDA-capable GPU and toolkit.
**Quick Start**: Clone the repo and run the early files (e.g., `0001_*`) to verify the ternary system works on your hardware.

## 🔍 Why This Exists
Created to solve problems in AI causality, this project uncovered an unexpected algorithmic pathway. It is released not as a finished solution, but as an **invitation** for the community to rigorously test a new spatial computing idea.

## 📄 More Information
visit https://cyborgunicorn.com.au/cuboids

## 📄 License
MIT Licence

