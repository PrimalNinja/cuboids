# Diffie-Hellman: A Computational Vulnerability Through Discretization?
## (Pollard's Rho, Shor's Algorithm and Constrained Geometry Wobble)

**Author:** Julian Cassin  
**Date:** February 2026  
**Status:** Initial Research Framework  

---

## Executive Summary

Diffie-Hellman (DH) key exchange is mathematically sound but computationally unsound. The protocol assumes infinite mathematical precision, but computers cannot compute infinite-precision mathematics. This gap between mathematical theory and computational reality could create an exploitable vulnerability.

When DH is implemented on finite hardware with bounded precision, the infinite circular modular space (mod p) is necessarily approximated as a finite polygon. This polygon has enumerable vertices. An attacker can potentially exploit this discretization to solve the discrete logarithm problem faster than claimed.

---

## 1. The Fundamental Problem

### 1.1 What DH Claims to Do

The Diffie-Hellman protocol states:

1. Both sides agree on a large prime **p** and base **g**
2. Alice picks private integer **a**, Bob picks private integer **b**
3. Alice computes **A = g^a mod p**, sends it to Bob
4. Bob computes **B = g^b mod p**, sends it to Alice
5. Both compute the shared secret: **S = g^(ab) mod p**

Security relies on: "It's easy to compute g^x mod p, but extremely hard to reverse the process (discrete logarithm problem)."

### 1.2 What DH Cannot Actually Do

The computation **g^a mod p** requires:
- Computing **g^a** (intermediate result)
- Applying **mod p** (reduction)

**The problem:** g^a is impossibly large.

For a 160-bit exponent:
- **2^160** is approximately 10^48 digits
- No computer can store or compute a number with 10^48 digits
- Therefore, **computers cannot actually compute g^a**

For a 2048-bit exponent:
- **2^2048** is approximately 10^616 digits
- The observable universe doesn't have enough storage

### 1.3 The Computational Shortcut

Since computers cannot compute g^a, cryptographic libraries use **modular exponentiation algorithms**:
- Compute g^a step-by-step
- Apply mod p at each step
- Keep intermediate results bounded to [0, p-1]
- Never let any intermediate exceed p

**This is not the same as computing g^a mod p.**

It is a computational approximation that produces the correct final result, but via a different path.

---

## 2. The Precision Problem

### 2.1 π as the Exemplar

Consider π (pi):

- π is deterministic (must fall on the circumference of a perfect circle)
- π is infinite (non-repeating decimal expansion)
- π can only be represented finitely on a computer

When stored in a computer:
- π ≈ 3.14159... (truncated to some precision)
- 2048 bits ≈ 617 decimal digits
- Computers store π to **at most 617 digits**

**Critical insight:** π itself is infinite, but computers force it into a finite representation. The truncation point creates a boundary condition.

### 2.2 Diffie-Hellman as a π Problem

The modular space mod p has the same structure as π:

- **π is deterministic** because it must fall on a circle's circumference
- **The modular space is deterministic** because all values must fall within a circular (modular) geometry

But computers force both into finite precision:
- π → truncated to 617 digits
- Modular space mod p → represented as a finite field with p-1 elements

**Possible vulnerability:** Both are infinite mathematical objects forced into finite computational spaces.

---

## 3. From Infinite Circle to Finite Polygon

### 3.1 The Geometric Transformation

An infinite geometric circle requires infinite precision to describe.

When a perfect circle (infinite precision) is represented on a computer with 2048-bit precision, it necessarily becomes a polygon with a finite number of vertices. This is not theoretical - it is how discretization works. The modular space mod p, which is geometrically circular, undergoes the same transformation. The resulting polygon is an objective, measurable structure.

The vertices are the discrete computational points where the algorithm can produce results.

### 3.2 How Many Sides?

The question that needs research:

**Given a 2048-bit modular space, how many discrete vertices/sides does the polygon have?**

This is NOT 2^2048 (the theoretical search space).

It's some number determined by:
- The computational precision (2048 bits)
- The reduction algorithm used (Montgomery, Barrett, etc.)
- The granularity of how mod p is computed

### 3.3 The Attack Vector

If the polygon has fewer sides than 2^2048, then:
- Enumeration becomes feasible on hardware that can enumerate the sides
- Pattern recognition can identify which side A = g^a mod p lands on
- Binary search can narrow down a through successive polygon partitions
- Fractional cascading can search p/2, p/4, p/8... reducing the search space exponentially

---

## 4. The Computational Reality

### 4.1 What's Actually Stored and Computed

When you implement DH:

```
p = 617-digit prime (approximately 2^2048)
g = small integer (2, 5, etc.)
a = 160+ bit integer
A = g^a mod p (result: 617-digit integer)
```

**The computer stores:**
- p: ~256 bytes
- a: ~32 bytes
- A: ~256 bytes

**The computer computes:**
- Modular exponentiation algorithm
- Multiple reduction steps
- All intermediate results bounded by p

**The computer never stores:**
- g^a (impossible: 10^616 digits)
- The "true" untruncated result

### 4.2 The Approximation Artifacts

Different cryptographic libraries implement modular exponentiation differently:
- OpenSSL uses one reduction method
- GMP uses another
- Python uses another

Each produces the mathematically identical result A, but via different **computational paths**.

These computational paths leave **side-channel signatures:**
- Timing variations
- Power consumption patterns
- Cache access patterns
- Memory bandwidth usage

An attacker measuring these signatures can deduce a, even though A itself is mathematically exact.

---

## 5. The Polygon Vertex Hypothesis

### 5.1 Formal Statement

**Hypothesis:** The discrete logarithm problem in a 2048-bit DH implementation is solvable in less than 2^2048 time because:

1. The infinite circular modular space (mod p) is approximated as a finite polygon
2. The polygon has a bounded number of vertices (V)
3. V is determined by computational precision and algorithm design, not by p itself
4. **V << 2^2048** (much less than the theoretical space)
5. An attacker can enumerate or search the polygon's vertices to find a

### 5.2 Why This Hasn't Been Published

Possible reasons:

- **Computational infeasibility with current hardware:** Even if the polygon has fewer vertices, the number might still be 10^600, making it impractical
- **Lack of formalization:** The geometric intuition is strong, but a rigorous mathematical proof is missing
- **Cryptographic community assumptions:** The field assumes hardness is proven; computational approximation artifacts are not treated as threats
- **Novelty:** This specific angle (π-inspired, geometry-based discretization) may not have been explored in this form

---

## 6. The Prime Distribution Angle

### 6.1 Prime Clustering and Geometric Alignment

Research has established:
- Primes cluster in patterns (twin primes, prime gaps)
- Prime density follows the Prime Number Theorem
- No known formula predicts the next prime

**New hypothesis:** Prime clusters may align with the polygon vertices of the modular space.

If true:
- Primes don't distribute randomly; they follow geometric constraints
- The modular space (defined by prime p) has geometric structure determined by p's position in the prime sequence
- The polygon vertices are **not uniform**; they cluster where primes cluster

### 6.2 Convergence to Circumference

Just as π's digits must converge to a circle's circumference:
- Primes may be forced to align with geometric structures
- Prime clustering may follow a "Law of the Wobble" (the pattern forcing convergence)
- Finding that law could simultaneously explain prime distribution AND break DH

---

## 7. Research Directions

### 7.1 Immediate Research Questions

1. **What is V (the polygon vertex count) for a 2048-bit modular space?**
   - Theoretical calculation required
   - Need to formalize "polygon discretization"
   - Likely involves reduction algorithm analysis

2. **Can primes be geometrically mapped to modular space vertices?**
   - Analyze prime distribution within mod p
   - Test for clustering at specific vertices
   - Compare with π's digit distribution

3. **Is there a "Law of the Wobble" for primes?**
   - Find the pattern forcing prime convergence
   - Apply to modular space geometry
   - Test against cryptographic parameters

4. **Can side-channel attacks on modular exponentiation be made practical?**
   - Measure computational signatures
   - Correlate with polygon vertex positions
   - Build timing/power analysis attacks

### 7.2 Required Expertise

- **Number theorists:** Formalize the geometric constraints
- **Cryptanalysts:** Design attacks exploiting polygon discretization
- **Computer architects:** Analyze computational paths in modular exponentiation
- **Information theorists:** Calculate effective search space reduction

### 7.3 Proof of Concept

A small-scale PoC would involve:

1. Choose a small prime p (e.g., 2^32 or 2^64)
2. Map the modular space as a geometric circle
3. Discretize it to see the polygon structure
4. Count vertices
5. Test if discrete log can be solved faster by exploiting vertices
6. Scale to larger p values

---

## 8. Implications

### 8.1 For Current Cryptography

- **DH/RSA/ECDH:** All rely on computational hardness of discrete log or factorization
- **Vulnerable to:** Attacks exploiting computational discretization
- **Timeline:** Unknown (depends on formalization and practical exploitability)
- **Urgency:** Post-quantum cryptography migration is correct decision

---

## 9. Pollard's Rho in Reverse: Constrained Geometry Wobble

### 9.1 What Pollard's Rho Assumes

Pollard's rho algorithm solves discrete log by:

1. Creating a pseudo-random walk through the modular space
2. Detecting when the walk enters a cycle
3. Using the cycle to compute the discrete log

**Time complexity:** O(√p) in the best case

**Fundamental assumption:** The walk is random, and the cycle is discovered probabilistically through collision detection.

The ρ diagram visualizes this: a seemingly random trajectory that eventually loops back on itself.

### 9.2 The Geometric Realization

But what if the "random walk" is not random at all?

What if the walk is a **wobble constrained by the polygon's geometry**?

Just as:
- π is deterministic (forced to fall on the circumference)
- Angles are deterministic (constrained by geometry)
- Side lengths are deterministic (calculated, not searched)

**The discrete log walk would also be deterministic:**
- a maps to a specific angle on the polygon
- That angle determines a specific vertex position
- The vertex position reveals a directly

### 9.3 From Probabilistic Search to Deterministic Calculation

**Current approach (Pollard's Rho):**
```
Search for the cycle
→ Find a collision
→ Compute discrete log from collision
→ Time: O(√p)
```

**Hypothetical geometric approach:**
```
Calculate the angle where a must fall
→ Identify the polygon vertex at that angle
→ Read the discrete log from vertex position
→ Time: O(?)  [potentially much faster]
```

The difference could be profound:

- **Pollard's rho searches for structure** it doesn't understand
- **Geometric calculation exploits structure** it does understand

### 9.4 The Missing Formula

The key insight is this:

**If the polygon vertices are determined by geometric law (like π's digits), then there exists a formula:**

```
Given: A = g^a mod p
       (A is a point on the circle/polygon)

Find: The angle θ where a must reside

Using: The geometric constraints that force a onto the polygon

Calculate: a = f(θ, p, g)

Where f() is the inverse geometric transformation
```

This is not polynomial interpolation or numerical approximation.

**This is geometric inversion.**

### 9.5 Why Pollard's Rho Works (and What It Misses)

Pollard's rho is **empirically efficient** because:
- The walk does eventually cycle (finite space guarantee)
- Cycle detection is fast (O(√p) collisions on average)

But Pollard's rho is **theoretically incomplete** because:
- It treats the cycle as accidental
- It doesn't explain *why* the cycle occurs at that specific location
- It doesn't leverage the geometric reason the cycle exists

**If the cycle is geometrically determined, a direct calculation would bypass the search entirely.**

### 9.6 The Research Question

**"Can we invert the polygon's geometry to calculate a directly, rather than discovering it through probabilistic walk-and-cycle?"**

This would require:

1. **Formalizing the geometric law** that constrains exponents to polygon vertices
2. **Deriving the inverse formula** (angle → exponent)
3. **Computing the formula** in less time than O(√p)
4. **Testing the approach** on small primes and scaling to cryptographic sizes

### 9.7 Connection to Shor's Algorithm

Peter Shor's quantum algorithm also works by **finding hidden structure:**
- Shor finds the period of a modular exponentiation function
- The period reveals the discrete log
- The period is deterministic (based on the group structure)

**Your hypothesis is similar:**
- The polygon structure is deterministic (based on geometry)
- The structure determines where a falls
- That position reveals the discrete log

Both approaches recognize that **DH has hidden structure that can be exploited if made explicit.**

The difference:
- Shor uses **quantum interference** to find the period
- Your approach uses **geometric calculation** to find the position

### 9.8 Why This Hasn't Been Explored

Cryptographers typically approach discrete log as:
- A **search problem** (how fast can we find a?)
- An **algebra problem** (can we solve the equation?)
- A **number theory problem** (what properties of p help?)

They rarely approach it as:
- A **geometry problem** (what does the geometry tell us?)
- A **constraint satisfaction problem** (where is a forced to be?)
- An **inversion problem** (can we reverse the geometric mapping?)

Pollard's rho is state-of-the-art for search-based approaches.

**But search might be the wrong strategy if the answer can be calculated.**

### 9.9 Immediate Test

For a small prime (e.g., p = 23, as in the DH toy example):

1. Compute all values: g^x mod p for x in [0, p-2]
2. Visualize them geometrically on a circle
3. Identify the polygon vertices
4. Measure the angles between vertices
5. Test if those angles follow a pattern based on π
6. Attempt to predict the next vertex position without enumeration
7. Compare Pollard's rho runtime with geometric calculation runtime

If the geometric approach outperforms Pollard's rho even at small scale, it suggests the hypothesis has merit.

---

## 10. Conclusion

Diffie-Hellman is mathematically sound but computationally unsound.

The protocol assumes:
1. Infinite-precision mathematics
2. Infinite search space
3. Unobservable computation

Computers provide:
1. Finite-precision approximations
2. Bounded computational spaces
3. Observable computational paths

This gap is not a minor implementation detail. It's a **possible structural vulnerability** in any cryptographic system based on computational hardness.

A possible vulnerability could be:
- **Theoretical:** The infinite circle becomes a finite polygon
- **Computational:** The polygon has enumerable vertices
- **Exploitable:** If the vertex count is significantly less than 2^2048

**The next step:** Formalize the polygon vertex hypothesis and test it against real cryptographic parameters.

---

## References & Further Reading

### Known Research (Cited)
- Discrete Logarithm Problem: Wikipedia, IACR eprints
- Baby-step Giant-step: Shanks' algorithm for discrete log in O(√p) time
- Pollard's rho: Probabilistic discrete log algorithm
- Polynomial Approximations of DL: Research on complexity bounds
- Finite Fields: Geometric interpretation as evenly-spaced points on unit circle

### Missing from Literature
- **Computational discretization of modular space as exploitable polygon**
- **Vertex enumeration for solving DH in sub-exponential time**
- **Geometric alignment of primes with polygon vertices**
- **"Law of the Wobble" for prime distribution and modular geometry**

---

## Appendix: Key Questions Unresolved

1. **What is the mathematical definition of "polygon discretization" in modular arithmetic?**
2. **How many vertices exist in a 2048-bit modular polygon?**
3. **Is V polynomial in the bit-size of p?**
4. **Can primes be predicted by understanding the wobble?**
5. **Does the answer change for elliptic curves vs. finite fields?**
6. **How does π's structure relate to prime distribution?**
7. **Can the vertex positions be calculated without enumeration?**
8. **What is the computational cost to map the polygon?**

---

**This document represents a research framework, not proven claims. The hypothesis requires rigorous mathematical development and empirical testing.**