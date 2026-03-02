# Cuboids & Nonoids for 3D Wave-Based Quantum Algorithms

**Author:** Julian Cassin  
**Date:** 2026-01-31  

## Overview

Cuboids and nonoids are 3D logical primitives designed to simplify reasoning about **spatially propagating states**. They offer a **human-comprehensible abstraction** for quantum programming, allowing us to conceptualize wave interference, superposition, and entanglement in three dimensions.

This document explores their potential as a programming model for quantum computers, particularly for **wave-like algorithms**, where spatial relationships govern computational propagation rather than discrete sequential instructions.

---

## 1. Definitions

### 1.1 Cuboid

- A 3D volumetric primitive.
- Comprised of:
  - **6 faces**
  - **12 edges**
  - **8 vertices**
  - **54 internal cells**  
- Total datapoints: **80**
- Useful for representing **volumetric spatial interactions**.

### 1.2 Nonoid

- Logical primitive with 9 planes (conceptually like 9 chessboards).
- Each plane has 9 points: 3×3 grid.
- Encodes **causal chains** of possibilities:
  - Eliminates invalid options
  - Cascades remaining “good” options to a primary plane
- Total datapoints: **111**
- Useful for **quantum-like superposition reasoning** and **constraint propagation**.

---

## 2. Conceptual Mapping to Quantum Computing

| Quantum Concept           | Nonoid/Cuboid Analogy                     |
|---------------------------|------------------------------------------|
| Qubit amplitude           | Cell value (-1, 0, +1)                   |
| Superposition             | Multiple planes/propagated states        |
| Interference              | Overlap and cancellation of cell values  |
| Entanglement              | Coupled planes or cuboid adjacency       |
| Measurement / collapse    | Selecting final consistent configuration |
| Quantum gates             | Propagation rules / neighbor interactions|

**Key Insight:**  
Nonoids allow **propagation-based reasoning**, enabling humans to “program waves” without directly manipulating matrices or symbolic gates.

---

## 3. Algorithmic Principles

1. **Spatial Propagation**
   - Each cell influences neighbors based on defined propagation rules.
   - Simulates interference patterns naturally.

2. **Constraint Elimination**
   - Nonoids can prune invalid states early.
   - Reduces search space akin to quantum amplitude amplification.

3. **Superposition Approximation**
   - Each plane represents a layer of possibilities.
   - Multi-plane interactions approximate **parallel amplitudes**.

4. **Collapse Simulation**
   - After propagation, extract a **consistent final state**.
   - Maps to the quantum measurement operation.

---

## 4. Example: Nonoid-Based Chess Solver (Quantum-Inspired)

1. **Initialize**: 9 planes representing potential moves.
2. **Propagate**: Apply move constraints across planes.
3. **Eliminate**: Remove invalid or dominated moves.
4. **Cascade**: Transfer viable moves to primary plane.
5. **Measure**: Select a final move that satisfies global constraints.

> Conceptually mirrors **quantum search algorithms** like Grover's search but in a **human-tractable, spatial representation**.

---

## 5. Potential Applications

- **Quantum Optimization**
  - Traveling Salesman, Knapsack, Scheduling
  - Nonoid planes approximate amplitude amplification.

- **Wave-Based Simulation**
  - Material simulations
  - Electromagnetic or fluid wave propagation

- **Quantum Machine Learning**
  - Map neural amplitudes to cuboid or nonoid cells
  - Enable interference-based classification

- **Algorithm Visualization**
  - Provide intuitive interfaces for quantum algorithms
  - Reduce dependence on matrix algebra

---

## 6. Advantages

- **Human-Comprehensible**
  - Easier to reason about than pure linear algebra.
- **Spatially Explicit**
  - Exploits adjacency and structure directly.
- **Parallelizable**
  - Each plane/cuboid can be propagated independently.
- **Hardware-Compatible**
  - Maps naturally to **3D quantum arrays or GPU-like quantum simulators**.

---

## 7. Research Directions

1. **Define Nonoid Propagation Rules**
   - Explore interference, entanglement, and cancellation models.

2. **Mapping to Qubits**
   - Investigate how cuboid/nonoid abstractions correspond to physical qubit states.

3. **Simulation on GPUs**
   - Use persistent kernel / tri-sword techniques to simulate 3D wave propagation efficiently.

4. **Optimization of Shape Sizes**
   - Adaptive cuboid/nonoid scaling for large problem spaces.

5. **Integration with Quantum Hardware**
   - Directly map planes to qubit registers or quantum-dot arrays.

6. **Visualization Tools**
   - 3D interactive interfaces for debugging and exploring superposition patterns.

---

## 8. Conclusion

Cuboids and nonoids provide a **novel, spatially-grounded abstraction** for programming quantum computers. By reasoning in terms of **3D wave propagation**, humans can intuitively design algorithms that leverage interference, superposition, and entanglement — without relying entirely on symbolic linear algebra.

This framework has potential to **accelerate quantum programming adoption** and serve as a bridge between **classical spatial reasoning** and **quantum computation**.

---

## 9. References & Inspiration

1. Julian Cassin, *Tri-Sword Architecture and Sovereign GPU Principles*, 2025  
2. Quantum Algorithms: Grover, Shor, and Amplitude Amplification  
3. 3D Cellular Automata and Wave Propagation Models  
4. Constraint Propagation in Artificial Intelligence (CSPs)  
