---
uri: phosphene.canon/jaccanon
owner: edphos
status: active
license: BSD-3-Clause
lineage:
  - order-relations
  - phosphene.equilibria
  - monograph-review
updated: 2026-11-10
---

# `jaccanon` — The Jacobian Canon
## Canonical Numerical Methods for Vector Differentiation, Matrix Decomposition, and Non-Equilibrium Flows

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Architecture: Diagnostic Contract](https://img.shields.io/badge/architecture-DFT-success)](docs/diagnostic-contract.md)

---

## 1. The tension between algebra and implementation

Abstract algebra has clean, coordinate-free invariants: morphisms compose, functors preserve structure, partially ordered sets formalize causal lineages, and symmetric/antisymmetric operators cleanly partition vector spaces.

**However, numerical algorithms approximate.**

When implementing numerical methods across Python libraries (`scipy`, `numpy`, `jax`, `casadi`), three fundamental structural tensions emerge immediately:

1. **Continuous Algebra vs. Finite-Precision Floating Point (IEEE 754):**
   In algebra, the symmetric operator $S = \frac{1}{2}(J + J^T)$ is exactly self-adjoint ($S = S^T$). In IEEE 754 float64, non-associativity of addition and roundoff noise produce spurious asymmetries that break downstream solvers expecting exact symmetry.
2. **Order Relations vs. Total Ordering in Memory:**
   In algebra, poset pre-orders, lattices, and causal partial orders ($\prec$) permit incomparability ($a \not\prec b \land b \not\prec a \land a \neq b$). In standard numerical routines (sorting, indexing, optimization bounds), algorithms force artificial total orders or arbitrary tie-breaking that leak into scientific conclusions.
3. **Silent Cancellation and the Floating-Point U-Curve:**
   In pure calculus, $\lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ approaches the exact derivative. On physical silicon, subtractive cancellation destroys up to 8–10 digits of significance as $h \to \epsilon_{\text{mach}}$, while truncation error dominates when $h$ is large.

`jaccanon` makes these structural tensions **legible and boundable under the ethos of scientific review and platform testability**.

In scientific computing, authors and domain researchers operate in the domain of *Discovery*—proposing novel inferential claims and forward models. The platform and review layer operates in the domain of **underwriting the reliability and insurance of those claims**. Underwriting establishes the quantified validity of computational results before acceptance: defining the explicit operational envelope where reliance holds, pricing proof obligations against failure severity, and restricting coverage strictly to stated, empirically verified regimes.

`jaccanon` realizes this underwriting role through **Design for Testability (DFT)**: providing **canonical numerical methods** with an explicit diagnostic contract that quantifies asymptotic truncation error bounds, measures subtractive cancellation loss, and certifies matrix condition stability at execution time.

---

## 2. The Dual Senses of "Canonical"

`jaccanon` operates at the exact bridge between the algebraic and engineering definitions of *canonical*:

```
                        THE CANONICAL BRIDGE
                                 │
     ALGEBRAIC SENSE             │             NUMERICAL SENSE
    (Canonical Form)             │         (Canonical Implementation)
                                 │
Unique normal representation     │ Vetted, hermetic primitive
modulo coordinate/gauge freedom. │ with proven error envelopes.
   e.g. J = S + A, <S, A>_F = 0   │    e.g. STDD + Hypothesis tests
                                 │
                                 ▼
                     Canonical Numerical Methods
              (Mathematical truth on finite machines)
```

| Domain | Definition | Role in `jaccanon` | Failure Mode Without the Other |
|---|---|---|---|
| **Algebraic Sense**<br>*(Canonical Form)* | *"The form, usually simplest or most symmetrical, to which, without loss of generality, all functions of the same class can be reduced."* | Unique normal forms: Jordan/Schur, SVD, and Helmholtz matrix splitting ($J = S + A$). | Abstract pencil-and-paper theorem without a verifiable numerical method. |
| **Numerical Sense**<br>*(Canonical Implementation)* | The vetted, tested, reproducible reference primitive that becomes the platform foundation. | Functional core, zero I/O, immutable models (`phosphene.core.models.InvariantStatic`), STDD property suites (`Hypothesis`). | Fragile bag of ad-hoc heuristics and arbitrary tolerances without structural grounding. |

---

## 3. System Architecture: The 4-Layer Computational Pipeline

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ Layer 1: Primal Vector Field                                     │
  │ f: R^n -> R^m (Dynamical flow, trait transmission, payoff field) │
  └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ Layer 2: Differentiation Primitives (jaccanon.prims)             │
  │ Atomic Operators: JVP, VJP, CS-Stencil, Richardson, CPR          │
  │ Observable Output: JacobianResult (Matrix, kappa, Truncation)    │
  └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ Layer 3: Helmholtz Normal Splitting (jaccanon.decomposition)     │
  │ Matrix Partition: J = S + A,  <S, A>_F = 0                       │
  │ Asymmetry Ratio: rho = ||A||_F / ||S||_F                         │
  └──────────────────────────────────────────────────────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼ (rho < eps_sym)                 ▼ (rho >= eps_sym)
  ┌───────────────────────────────┐ ┌────────────────────────────────┐
  │ Layer 4A: Conservative Solver │ │ Layer 4B: Non-Conservative Flow│
  │ Gradient Descent / Potential  │ │ Korpelevich Extragradient      │
  │ Monotone descent on Phi(x)    │ │ Two-step prediction-correction │
  └───────────────────────────────┘ └────────────────────────────────┘
```

---

## 4. The Diagnostic Contract & Design for Testability (DFT)

**Design for Testability (DFT)** establishes testability as an intrinsic structural property of every operator. Operators embed diagnostic observability, controllable perturbation parameters, and invariant verification directly into their interface.

In this architecture, the platform underwrites computational claims across three distinct vertices:
1. **Mathematics:** Verifying operator invariants, algebraic dualities, and spectral limits ($J = S + A$, $\langle S, A \rangle_F = 0$, pushforward/pullback duality).
2. **Software (The Implementation):** Ensuring the implementation realized in code exposes its precision boundaries, conditioning, and asymptotic convergence rates without silent fallback heuristics.
3. **Data / Numerical Inputs:** Reconciling perturbations, evaluation budgets, and IEEE 754 floating-point constraints against empirical stability basins.

Every differentiation primitive returns an immutable `JacobianResult` based on `phosphene.core.models.InvariantStatic` (`frozen=True`, `extra='forbid'`), fulfilling the diagnostic contract:
- **Matrix Condition Number $\kappa(J)$:** Computed via singular values, warning when $\kappa(J) \ge 10^8$.
- **Truncation Error Estimate:** Theoretical asymptotic $O(h^p)$ error bound.
- **Roundoff & Cancellation Tracking:** Tracks $O(\epsilon_{\text{mach}} / h)$ and estimates decimal digits destroyed by subtractive cancellation.
- **Pathology Flags:** Explicit flags for `RANK_DEFICIENT_OR_SINGULAR`, `UPPER_BOUND_HIT`, and `CUSP_DETECTED`.

See [docs/diagnostic-contract.md](docs/diagnostic-contract.md) for full details on the three pillars of numerical DFT and the underwriting model.

---

## 5. Intellectual Genealogies

`jaccanon` bridges classical mathematical traditions and structural operator lineages into canonical numerical methods (see [docs/genealogies.md](docs/genealogies.md)):

### Foundational Mathematical Lineages
1. **Order-Theoretic Algebra & The Epistemic Gap:** Preserving partial-order incomparability and lattice structures without forcing artificial total orderings onto IEEE 754 floating-point representations.
2. **Operator Decomposition & Discrete Helmholtz Splitting:** Decomposing Jacobians ($J = S + A$) to isolate dissipative potential sinks ($S$) from solenoidal rotational torques ($A$), automating solver routing between conservative gradient descent and non-conservative Korpelevich extragradient projection.
3. **Critical Phenomena & Relaxation Dynamics:** Grounded in formal relaxation dynamics from statistical mechanics and catastrophe theory (Ising $\to$ Landau $\to$ Thom Cusp), modeling multi-scale non-equilibrium flows.

### Classical Differentiation Genealogies

| Historical Lineage | Canonical Primitive (`jaccanon.prims`) | External Reference Target |
|---|---|---|
| **Empirical / Finite Differences**<br>*(Euler, Richardson, Wilkinson, D'Errico)* | `jacobian_central_3point`<br>`RichardsonTableau`<br>`ComplexStepAnalytic` | `scipy.optimize._numdiff`<br>`numdifftools.Jacobian`<br>`jacobi.jacobi` |
| **Structural AD & Graph Coloring**<br>*(Clifford, Rall, Griewank, Diehl)* | `PureDualNumber`<br>`WengertTapeAdjoint`<br>`CPRColumnColoring` | `CasADi` (SX/MX Graph AD)<br>`scipy.sparse.group_columns` |
| **Compiler IR & Vectorized Transforms**<br>*(Frostig, He, Bradbury)* | `FunctionalJVP`<br>`FunctionalVJP`<br>`DimensionRouter` | `jax.jacfwd` / `jax.jacrev`<br>`torch.func` |

---

## 6. Empirical Reproductions & Claims Evaluation Suite

`jaccanon` includes a method verification test suite (`tests/test_reproductions.py`) that systematically benchmarks and evaluates the mathematical claims of external engines:

| Method / Library Tested | Stated Claim | Empirical Result | Stress / Breakdown Envelope |
|---|---|---|---|
| **SciPy `_numdiff` (2-point)** | $O(\sqrt{\epsilon_m})$ accuracy | **Verified** ($2.49 \times 10^{-8}$) | Subtractive cancellation destroys **8 digits** of precision out of 16. |
| **SciPy `_numdiff` (3-point)** | $O(\epsilon_m^{2/3})$ accuracy | **Verified** ($2.28 \times 10^{-11}$) | Reaches 11-digit ceiling; **5 digits** lost to cancellation. |
| **Complex-Step Analytic** | Machine precision $\sim 10^{-16}$ at $h=10^{-30}$ | **Verified (Holomorphic)** | **Total failure on non-holomorphic operations**: $\frac{d|x|}{dx}$ at $x=1$ yields $0.0$ instead of $1.0$ (100% error). |
| **Modern AD (`torch.func`)** | Exact derivatives to machine precision | **Verified (Smooth)** | **Silent cusp failure**: non-differentiable points ($x=0$ on $|x|$) return subgradient $0.0$ with zero pathology warnings. |

See full empirical audit in [`docs/empirical-claims-reproduction-audit.md`](docs/empirical-claims-reproduction-audit.md).

---

## 7. Quickstart

```python
import numpy as np
from jaccanon.prims.empirical import jacobian_central_3point
from jaccanon.decomposition.helmholtz import decompose_matrix_sa

# 1. Define a nonlinear vector field
def f(x: np.ndarray) -> np.ndarray:
    return np.array([
        x[0]**2 + 2.0 * x[1],
        -2.0 * x[0] + x[1]**2
    ])

# 2. Differentiate with full observability
res = jacobian_central_3point(f, np.array([1.0, 1.0]))
print(f"Condition Number kappa(J): {res.condition_number:.2f}")
print(f"Estimated Truncation Error: {res.estimated_truncation_error:.2e}")
print(f"Estimated Cancellation Loss: {res.loss_of_significance_digits:.1f} digits")

# 3. Decompose into symmetric (gradient) and antisymmetric (curl) components
decomp = decompose_matrix_sa(res.matrix)
print(f"Asymmetry ratio (rho): {decomp.asymmetry_ratio:.3f}")
print(f"Regime: {decomp.regime}")
print(f"Recommended Solver: {decomp.recommended_solver}")
# Output:
# Asymmetry ratio (rho): 1.000
# Regime: non_conservative
# Recommended Solver: extragradient_projection
```

---

## 8. Scientific Test-Driven Development (STDD) & Invariant Contracts

All primitives in `jaccanon` are validated against invariant contracts via `Hypothesis` property-based testing:

1. **Taylor Remainder Invariant:** First-order convergence test:
   $$\lim_{\|v\| \to 0} \frac{\|f(x + v) - f(x) - J \cdot v\|}{\|v\|} = 0, \quad \text{slope of } \log R(v) \text{ vs } \log \|v\| \to 2.0$$
2. **Adjoint Duality Invariant:** Pushforward / pullback duality across covector spaces:
   $$\langle u, \text{JVP}(f, x, v) \rangle \equiv \langle \text{VJP}(f, x, u), v \rangle, \quad \forall u \in \mathbb{R}^m, v \in \mathbb{R}^n$$
3. **Helmholtz Orthogonality Invariant:**
   $$\langle S, A \rangle_F = \text{tr}(S^T A) \equiv 0 \quad \text{and} \quad S + A \equiv M$$
4. **Holomorphic Step Consistency:** Complex-step stencils match analytic dual-number evaluation to within $5 \cdot \epsilon_{\text{mach}}$.

---

## 9. Citation

```bibtex
@software{jaccanon2026,
  author = {Edward Phillips (Phosphene)},
  title = {jaccanon: Canonical Numerical Methods for Vector Differentiation, Matrix Decomposition, and Non-Equilibrium Flows},
  year = {2026},
  url = {https://github.com/phosphene/jacobian-canon}
}
```
