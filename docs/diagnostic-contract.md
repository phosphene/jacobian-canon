---
title: The Diagnostic Contract and Design for Testability (DFT)
author: Ed Phil & Flow (@feelingflowingbot)
status: living
created: 2026-11-10
---

# The Diagnostic Contract and Design for Testability (DFT)

In scientific software design, **Design for Testability (DFT)** is derived from a fundamental platform ethos: **underwriting the reliability and insurance of inferential claims**.

### The Underwriting Analogy in Scientific Software
Authors and domain scientists operate in the domain of *Discovery*—proposing novel inferential claims, forward models, and empirical fits. The platform and review teams operate in the domain of **underwriting the reliability and insurance of those claims**.

Underwriting is the actuarial practice of assessing proposed risks prior to acceptance, establishing the operational envelope where coverage holds, and verifying empirical proof obligations. Applied to computational science:
- **The proposed risk:** The inferential claim, parameter estimate, or forward model seeking coverage.
- **Coverage:** Reliance—being cited, published, deployed, or used as a trusted foundation for downstream pipelines.
- **Underwriting:** The systematic evaluation verifying that the computational implementation faithfully realizes the stated mathematical and empirical assertions.
- **The proof obligation (premium):** The suite of invariant tests, condition budgets, and error-bound certifications that the operator must satisfy to obtain coverage.
- **The policy:** The bounded operational envelope—coverage applies to specifically verified parameter spaces and precision regimes.

Claims awaiting underwriting verification remain classified as provisional and uncertified.

### The Three Vertices of Underwriting
Every numerical operator in `jaccanon` is underwritten across three independent vertices:
1. **Mathematics:** The formal equations, algebraic dualities, and spectral decompositions ($J = S + A$, $\langle S, A \rangle_F = 0$, pushforward/pullback duality).
2. **Software (Implementation):** The executable implementation—ensuring algorithms explicitly surface condition numbers, error bounds, and convergence rates through typed return types.
3. **Data / Numerical Inputs:** The empirical perturbations, domain meshes, and IEEE 754 floating-point representations.

Because finite precision arithmetic introduces subtractive cancellation, truncation drift, ill-conditioning ($\kappa(J) \gg 10^8$), and non-differentiable cusps, `jaccanon` implements an explicit **Diagnostic Contract** across every primitive.

---

## 1. The Three Pillars of Numerical DFT

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NUMERICAL DFT PILLARS                           │
├────────────────────────────────────────────────────────────────────────┤
│ 1. OBSERVABILITY                                                       │
│    Every derivative evaluation yields an immutable JacobianResult       │
│    carrying condition numbers, truncation/roundoff budgets, and         │
│    pathology flags.                                                    │
│                                                                        │
│ 2. CONTROLLABILITY                                                     │
│    Step sizes, directional perturbation bases, checkpoint depths, and   │
│    coloring seeds are exposed as first-class controls.                 │
│                                                                        │
│ 3. INVARIANT VERIFICATION                                              │
│    Every routine is continuously tested against mathematical axioms    │
│    using property-based testing (Hypothesis) rather than fixed oracle   │
│    magic numbers.                                                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The `JacobianResult` Manifest

Rather than returning a naked NumPy array or tensor, all differentiation primitives return an immutable `JacobianResult` inheriting from the canonical `phosphene.core.models.InvariantStatic` (`frozen=True`, `extra='forbid'`):

```python
from phosphene.core.models import InvariantStatic

class JacobianResult(InvariantStatic):
    matrix: tuple[tuple[float, ...], ...]
    shape: tuple[int, int]
    method: str
    genealogy: str

    # Observability: Error bounds & Conditioning
    estimated_truncation_error: float   # O(h^p) bound
    estimated_roundoff_error: float     # O(eps_mach / h) bound
    loss_of_significance_digits: float  # Digits destroyed by cancellation
    condition_number: float             # SVD-based condition number kappa(J)

    # Complexity expenditure
    primal_evaluations: int             # Raw calls to f(x)
    directional_passes: int             # JVP or VJP sweeps
    memory_peak_bytes: int              # Peak memory usage
    floating_point_precision: str       # 'float64', etc.

    # Diagnostic warnings
    pathology_flags: tuple[str, ...]    # e.g., ('BOUNDARY_PROXIMITY',)
```

---

## 3. The Numerical U-Curve and Step Controllability

For empirical differentiation, error follows the classic Wilkinson U-curve:

$$\text{Total Error}(h) \approx C_{\text{trunc}} h^p + C_{\text{round}} \frac{\epsilon_{\text{mach}}}{h}$$

```
  Error log10(E)
        ▲
     -4 │ \  Truncation: O(h^2)                 /  Roundoff: O(eps / h)
        │  \                                   /
     -8 │   \                                 /
        │    \                               /
    -12 │     \                             /
        │      \           ★   ★           /
    -16 │       \__________★___★__________/   <── Optimal Step h_opt
        └────────────────────────────────────────► log10(h)
          -16   -14   -12   -10    -8    -6   -4   -2
```

`jaccanon` provides automated diagnostic probes that map this U-curve programmatically, proving whether an empirical step size resides in the stable basin or on the roundoff cancellation cliff.
