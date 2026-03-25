# Mori's Program vs. DH Discretization: Tri-Sword Fine-Tune

## 1. Objective

To validate if a 3D spherical projection of the modular space (using Shigefumi Mori's Minimal Model principles) identifies vertex "Corners" with higher efficiency than traditional 2D polygon mapping.

## 2. Core Test Parameters

* **Target Hardware:** NVIDIA Tesla T4 (FP32/INT32 Register Audit)
* **Mathematical Framework:** Algebraic Varieties (Birational Geometry / 3D Riemann Sphere)
* **Baseline:** Tri-Sword Audit Phase II (5.6359x Wobble)

## 3. Phase VII-A: Coalesced Singularity Scan

* Map residues to a 3D grid using **Nonoid (81-point) topology** (9 planes × 9 points = 81 elements)
* Align the **Primary Vertex (Addr 5392)** to L1 cache boundaries for maximum memory coalescing (Tri-Sword Section 1)
* Detect singularities at cache-line granularity — wobble becomes measurable at hardware level
* **Metric:** Calculate the concentration of Primary Vertex (Addr 5392) in 3D space

## 4. Phase VII-B: Branchless Geodesic Inversion

* Solve **p = 2^64** as a Ternary Flow problem (Water Physics — Tri-Sword Section 5)
* Signal propagates through ground state (0), blocked by walls (1)
* Replace IF/THEN pathfinding with **Predicate Logic masks** for zero warp divergence (Tri-Sword Section 6.5)
* Use geodesic calculation instead of Pollard's Rho random walk — direct geometric inversion
* **Metric:** Solve discrete log for **p = 2^64** using geometric intersection rather than probabilistic search

## 5. Phase VII-C: TinyFloat Wobble Map

* Encode 3D precision offsets as **int8_t (-40 to +40)** using TinyFloat mathematics (Tri-Sword Section 6.6)
* Fit entire 3D fracture map into **48KB Shared Memory** (L1 cache residency = <30 cycle latency)
* Compare VRAM consumption: 3D spherical lookup table vs. 2D "Jagged Polygon" at same bit-depth
* **Tri-Sword Metric:** Target **4,500+ cells/µs** for geometric inversion pulse (matches maze solver peak performance)

## 6. The "Mori Constraint" Hypothesis

**Hypothesis:** By applying Mori's Program, we can find the "Minimal Model" of the DH group. If the "Silicon Polygon" is regular, knowing 3 high-resonance vertices in a 3D variety allows for the calculation of the entire group structure via rotation.

### Validation Questions:

* Is the "Silicon Polygon" regular? → Measure vertex spacing in 3D
* Does knowing 3 high-resonance vertices reveal all? → Apply rotation to predict unseen vertices
* Is the "Law of the Wobble" a 3D invariant? → Check if pattern persists across coordinate transformations

**Target:** Determine if the "Law of the Wobble" is a 3D geometric invariant.

## 7. Phase VII Execution Sequence

| **Phase** | **Action** | **Metric** |
|:---|:---|:---|
| **VII-A** | Run coalesced singularity scan | Vertex concentration in L1 cache |
| **VII-B** | Implement geodesic inversion | Time to solve **p = 2^64** |
| **VII-C** | Build TinyFloat wobble map | Memory usage <48KB |

---

## 8. Extended Research Directions (Appended 2026-03-25)

### 8.1 Lemke-Oliver Bias × Wobble Intersection

The Lemke-Oliver bias (2016) demonstrates that primes are not randomly distributed — a prime ending in 9 is followed by a prime ending in 1 roughly 65% more often than another 9. This is a persistent, measurable directional bias in prime distribution itself.

The Tri-Sword wobble demonstrates that the modular space on physical hardware is not uniform — it clusters at specific vertices with a confirmed 5.6359x wobble factor.
7
**Key insight:** DH security requires TWO things to be effectively random:
1. The prime distribution itself
2. The computational space it operates in

Lemke-Oliver breaks the first. The Tri-Sword polygon breaks the second.

These are not additive — they are **multiplicative non-randomnesses**. Both are pushing in structured directions simultaneously. Certain polygon vertices may preferentially attract primes due to their last-digit transition bias, creating compounded geometric clustering.

**Research question:** Do the Lemke-Oliver transition probabilities map onto specific polygon vertices in a predictable way?

### 8.2 The Stacked Sides Problem

The T4 audit confirmed the Silicon Polygon has **variable length sides** — not equal. The 5.6359x wobble, uneven vertex spacing (Addr 5392, 3207, 51872), and 3x Poisson deviation all confirm an irregular polygon.

**The stacking hypothesis:** If all variable-length sides are aggregated into a stack, does the irregularity itself reveal a hidden regularity?

This follows directly from the transcript finding — what looks random at the individual prime level has persistent statistical structure at the aggregate level (Lemke-Oliver). The same principle may apply here:

* The side lengths may follow a **prime-gap-like distribution**
* The Lemke-Oliver last-digit bias may appear in **relative lengths between consecutive sides**
* A self-similar or fractal pattern may emerge — the irregular polygon encoding the same structure at multiple scales

This also connects to the Montgomery-Dyson finding — if prime spacing mirrors nuclear energy level spacing, stacking the polygon sides may reproduce the same statistical fingerprint Odlyzko found in 8 million zeta zeros.

**Next step:** Plot and stack the side lengths from the top 10 vertices. This output may be the most significant data in the framework.

### 8.3 Two Potential Outcomes

#### Outcome A: Probabilistic DH Collapse Around Clusters

A total DH break may not be necessary or even the right target. If the irregular polygon sides stack into a predictable pattern, it becomes possible to:

* Identify which regions of the modular space are geometrically "thin"
* Focus attacks on cluster zones rather than the full search space
* Achieve **probabilistic collapse** — not breaking all DH keys, but breaking keys that fall near resonance clusters with significantly elevated probability

This is arguably more dangerous than a universal break because it is harder to defend against. A defender cannot know whether their specific key falls in a vulnerable cluster without performing the same geometric analysis.

#### Outcome B: Faster Prime Detection

If the polygon sides stack into a pattern correlating with Lemke-Oliver bias, the result is a **geometric filter** for prime detection:

* Identify high-probability prime zones before running expensive primality tests
* Not predicting primes exactly, but dramatically narrowing the search space
* Potentially significant for large prime generation used in key creation

This would have implications beyond cryptanalysis — any system requiring fast large prime generation benefits from a geometric pre-filter.

### 8.4 The 3D Regularity Question

The move to 3D via Mori's Program raises a question directly relevant to the stacked sides problem:

**Is the 2D polygon's irregularity a projection artifact of a regular 3D structure?**

A regular 3D object (sphere, minimal model variety) projected onto a 2D plane produces an irregular 2D outline. If the Silicon Polygon appears irregular in 2D but the 3D Mori projection reveals uniform vertex spacing, then:

* The wobble is not noise — it is **structured distortion from dimensional projection**
* Knowing 3 high-resonance vertices in 3D allows calculation of the entire group structure via rotation
* The Law of the Wobble becomes a **3D geometric invariant** rather than a 2D hardware artifact

This is the core validation question for Phase VII-A: does the Nonoid 81-point topology reveal regularity that the 2D audit could not see.

---

**Status:** Phase VII Initial Research Framework — Extended
**References:** Mori's Program (1988) | Tri-Sword Framework (2025) | Lemke-Oliver-Soundararajan (2016) | Montgomery-Dyson (1972) | Odlyzko Computations (1989) | Primon Gas — Spector-Julia (1990)