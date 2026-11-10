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
## Reference Canon for Vector Differentiation, Matrix Decomposition, and Non-Equilibrium Flows

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Architecture: Phosphene DFT](https://img.shields.io/badge/architecture-DFT-success)](docs/architecture.md)

---

## 1. The tension between algebra and implementation

Abstract algebra has clean, coordinate-free invariants: morphisms compose, functors preserve structure, partially ordered sets formalize causal lineages, and symmetric/antisymmetric operators cleanly partition vector spaces.

**However, numerical algorithms approximate.**

Numerical methods in  Python libraries (`scipy`, `numpy`, `jax`, `casadi`), three structural tensions emerge immediately:

1. **Continuous Algebra vs. Finite-Precision Floating Point (IEEE 754):**
   In algebra, the symmetric operator $S = \frac{1}{2}(J + J^T)$ is exactly self-adjoint ($S = S^T$). In IEEE 754 float64, non-associativity of addition and roundoff noise produce spurious asymmetries that break downstream solvers expecting exact symmetry.
2. **Order Relations vs. Total Ordering in Memory:**
   In algebra, poset pre-orders, lattices, and causal partial orders ($\prec$) permit incomparability ($a \not\prec b \land b \not\prec a \land a \neq b$). In standard numerical routines (sorting, indexing, optimization bounds), algorithms force artificial total orders or arbitrary tie-breaking that leak into scientific conclusions.
3. **Silent Cancellation and the Floating-Point U-Curve:**
   In pure calculus, $\lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ approaches the exact derivative. On physical silicon, subtractive cancellation destroys up to 8–10 digits of significance as $h \to \epsilon_{\text{mach}}$, while truncation error dominates when $h$ is large.

`jaccanon` is built to resolve this tension. It does not treat the Jacobian as an opaque black-box array; it provides an **executable reference canon** with explicit invariant contracts and complete observability of numerical error budgets upfront.

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
                     Executable Reference Canon
              (Mathematical truth on finite machines)
```

| Domain | Definition | Role in `jaccanon` | Failure Mode Without the Other |
|---|---|---|---|
| **Algebraic Sense**<br>*(Canonical Form)* | *"The form, usually simplest or most symmetrical, to which, without loss of generality, all functions of the same class can be reduced."* | Unique normal forms: Jordan/Schur, SVD, and Helmholtz matrix splitting ($J = S + A$). | Abstract pencil-and-paper theorem without an executable computational engine. |
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

Borrowing from safety-critical hardware engineering, `jaccanon` enforces **Design for Testability (DFT)**: internal states, truncation errors, and structural invariants must be observable through the API itself.

Every differentiation primitive returns an immutable `JacobianResult` based on `phosphene.core.models.InvariantStatic` (`frozen=True`, `extra='forbid'`):
- **Matrix Condition Number $\kappa(J)$:** Computed via singular values, warning when $\kappa(J) \ge 10^8$.
- **Truncation Error Estimate:** Theoretical asymptotic $O(h^p)$ error bound.
- **Roundoff & Cancellation Tracking:** Tracks $O(\epsilon_{\text{mach}} / h)$ and estimates decimal digits destroyed by subtractive cancellation.
- **Pathology Flags:** Explicit flags for `RANK_DEFICIENT_OR_SINGULAR`, `UPPER_BOUND_HIT`, and `CUSP_DETECTED`.

---

## 5. Intellectual Genealogies & Phosphene Foundations

`jaccanon` bridges four classical mathematical lineages with three internal foundational test benches developed within the Phosphene platform and the Research Stable:

### Foundational Phosphene Lineages
1. **`order-relations` & The Epistemic Gap:** Codifies the boundary between abstract posets and finite floating-point execution (`work/flow/myth-study/wiki/concepts/order-relations-and-numerical-methods.md`).
2. **The Foundry (`phosphene.equilibria`):** Implements the discrete Helmholtz decomposition ($J = S + A$) connecting dissipative potential sinks ($S$) and solenoidal rotational torques ($A$), automating routing between conservative gradient descent and non-conservative Korpelevich extragradient projection.
3. **`monograph-review` (Relaxation Dynamics):** Grounded in the formal chain from Ising (1925) $\to$ Landau (1937) $\to$ Cusp (Thom 1972) $\to$ Relaxation dynamics ($d\rho/dt = -k_1(\rho-\rho_1) - k_2(\rho-\rho_2)$).

### Classical Differentiation Genealogies

| Historical Lineage | Canonical Primitive (`jaccanon.prims`) | External Reference Target |
|---|---|---|
| **Empirical / Finite Differences**<br>*(Euler, Richardson, Wilkinson, D'Errico)* | `jacobian_central_3point`<br>`RichardsonTableau`<br>`ComplexStepAnalytic` | `scipy.optimize._numdiff`<br>`numdifftools.Jacobian`<br>`jacobi.jacobi` |
| **Structural AD & Graph Coloring**<br>*(Clifford, Rall, Griewank, Diehl)* | `PureDualNumber`<br>`WengertTapeAdjoint`<br>`CPRColumnColoring` | `CasADi` (SX/MX Graph AD)<br>`scipy.sparse.group_columns` |
| **Compiler IR & Vectorized Transforms**<br>*(Frostig, He, Bradbury)* | `FunctionalJVP`<br>`FunctionalVJP`<br>`DimensionRouter` | `jax.jacfwd` / `jax.jacrev`<br>`torch.func` |

---

## 6. Empirical Reproductions & Claims Evaluation Suite

`jaccanon` includes an executable reproduction test suite (`tests/test_reproductions.py`) that systematically benchmarks and evaluates the mathematical claims of external engines:

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

## 7. Scientific Test-Driven Development (STDD) & Invariant Contracts

All primitives in `jaccanon` are validated against invariant contracts via `Hypothesis` property-based testing:

1. **Taylor Remainder Invariant:** First-order convergence test:
   $$\lim_{\|v\| \to 0} \frac{\|f(x + v) - f(x) - J \cdot v\|}{\|v\|} = 0, \quad \text{slope of } \log R(v) \text{ vs } \log \|v\| \to 2.0$$
2. **Adjoint Duality Invariant:** Pushforward / pullback duality across covector spaces:
   $$\langle u, \text{JVP}(f, x, v) \rangle \equiv \langle \text{VJP}(f, x, u), v \rangle, \quad \forall u \in \mathbb{R}^m, v \in \mathbb{R}^n$$
3. **Helmholtz Orthogonality Invariant:**
   $$\langle S, A \rangle_F = \text{tr}(S^T A) \equiv 0 \quad \text{and} \quad S + A \equiv M$$
4. **Holomorphic Step Consistency:** Complex-step stencils match analytic dual-number evaluation to within $5 \cdot \epsilon_{\text{mach}}$.

---

## 8. Citation

```bibtex
@software{jaccanon2026,
  author = {Edward Philips (Phosphene)},
  title = {jaccanon: An Executable Reference Canon for Vector Differentiation, Matrix Decomposition, and Non-Equilibrium Flows},
  year = {2026},
  url = {https://github.com/phosphene/jacobian-canon}
}
```
