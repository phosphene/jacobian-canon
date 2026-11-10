---
title: Intellectual Genealogies of Vector Differentiation
author: Ed Phil & Flow (@feelingflowingbot)
status: living
created: 2026-11-10
---

# Intellectual Genealogies of Vector Differentiation

Numerical differentiation is not a single technique; it is a convergence of distinct intellectual traditions across analysis, numerical linear algebra, computational geometry, and programming language compilers.

`jaccanon` organizes these traditions into four reproducible lineages:

---

## Lineage 0: Mathematical Foundations & Structural Lineages

Before decomposing external differentiation algorithms, `jaccanon` is grounded in three foundational structural lineages:

### 1. Order-Theoretic Algebra & The Epistemic Gap
- **Core Invariant:** *Algebra does not calculate; numerical algorithms do.*
- **Epistemic Principle:** Continuous algebra assumes infinite precision, exact symmetry, and unconstrained partial orders. Finite-precision floating-point arithmetic (IEEE 754) introduces non-associativity, roundoff noise, and artificial total ordering.
- **Architectural Defense:** Rather than claiming to eliminate these tensions, `jaccanon` makes them legible and boundable: rigorous property-based testing (`Hypothesis`) and explicit condition budgeting ensure algebraic structures and partial incomparability are preserved without silent corruption.

### 2. Discrete Helmholtz Decomposition & Non-Conservative Operators
- **Core Invariant:** The Discrete Helmholtz Decomposition Theorem ($J = S + A$).
- **Physical Interpretation:**
  - $S = \frac{1}{2}(J + J^T)$: Dissipative sink dynamics (gradient flow toward potential minima).
  - $A = \frac{1}{2}(J - J^T)$: Solenoidal rotational curl (entropy-preserving circulating torques; prevents mode collapse in multi-agent game equilibria and non-equilibrium steady states).
- **Automated Routing:** Symmetrization metric $\rho = \|A\|_F / \|S\|_F < \epsilon_{\text{sym}}$ routes dynamically between conservative optimization and Korpelevich extragradient projection.

### 3. Critical Phenomena & Relaxation Dynamics
- **Core Invariant:** Formal relaxation chain from statistical mechanics and catastrophe theory: Ising (1925) $\to$ Landau (1937) $\to$ Cusp (Thom 1972) $\to$ Relaxation formula:
  $$\frac{d\rho}{dt} = -k_1(\rho - \rho_1) - k_2(\rho - \rho_2)$$
- **Role in `jaccanon`:** Establishes that dynamic equilibria are relaxation landscapes governed by energy potentials and rotational flows.

---

## Lineage 1: Empirical Analysis & Stencil Extrapolation
- **Key Thinkers:** Leonhard Euler, Lewis Fry Richardson, James H. Wilkinson, John D'Errico.
- **Foundational Idea:** Approximate limits via discrete sampling of black-box evaluations.
- **Key Techniques:**
  - Forward 2-point stencils ($O(h)$).
  - Central 3-point stencils ($O(h^2)$).
  - Richardson polynomial extrapolation (deferred approach to the limit, eliminating successive powers of $h^2$).
  - Lyness-Moler complex-step differentiation ($\text{Im}(f(x + ih))/h$, eliminating subtractive cancellation in holomorphic functions).
- **Core Trade-off:** High generality (works on black boxes), but constrained by the floating-point U-curve and $O(n)$ function evaluations.

---

## Lineage 2: Structural Automatic Differentiation (AD) & Wengert Tapes
- **Key Thinkers:** William Kingdon Clifford (dual numbers), Louis B. Rall, Andreas Griewank, Moritz Diehl.
- **Foundational Idea:** Decompose programs into directed acyclic graphs (DAGs) of elementary arithmetic operations and apply the chain rule algebraically.
- **Key Techniques:**
  - **Forward Mode:** Dual numbers $a + b\epsilon$ with $\epsilon^2 = 0$. Evaluates directional derivatives (JVP) alongside primal passes in $O(1)$ memory.
  - **Reverse Mode:** Wengert execution tape recording primal values, followed by reverse accumulation (VJP) in covector space. Evaluates gradients in $O(1)$ evaluations regardless of input dimension $n$.
- **Core Trade-off:** Exact derivatives without truncation error, but reverse mode incurs an execution tape memory footprint proportional to operation count.

---

## Lineage 3: Graph Compression & Matrix Sparsity
- **Key Thinkers:** Curtis, Powell, and Reid (CPR, 1974), Thomas F. Coleman, Jorge Moré, Alex Pothen.
- **Foundational Idea:** Real-world physical, biological, and dynamical Jacobians are sparse. Columns that do not share non-zero row indices can be evaluated simultaneously using a single perturbed seed vector.
- **Key Techniques:**
  - Distance-1 column coloring (CPR).
  - Distance-2 star coloring for bidirectional Jacobian accumulation.
  - Compressed accumulation and direct decompression.
- **Core Trade-off:** Reduces $n$ function calls to $\chi$ calls (where $\chi \ll n$ is the chromatic number of the Jacobian adjacency graph).

---

## Lineage 4: Compiler Intermediate Representation (IR) & Vectorized Transforms
- **Key Thinkers:** Theano team, Dougal Maclaurin (Autograd), Roy Frostig, Matthew Johnson, Chris Lattner.
- **Foundational Idea:** Treat differentiation as a higher-order program transformation at the compiler IR level rather than runtime tape tracing.
- **Key Techniques:**
  - Functional pushforward (`jvp`) and pullback (`vjp`) primitives.
  - Vectorizing maps (`vmap`) over batch dimensions.
  - Aspect-ratio dispatch: computing $J$ via `vmap(jvp)` when $n \le m$ (tall) or `vmap(vjp)` when $m < n$ (wide).
- **Core Trade-off:** Maximum hardware acceleration and kernel fusion, but requires pure functional transformations and AST/IR tracing.
