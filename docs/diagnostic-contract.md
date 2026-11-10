---
title: The Diagnostic Contract and Design for Testability (DFT)
author: Ed Phil & Flow (@feelingflowingbot)
status: living
created: 2026-11-10
---

# The Diagnostic Contract and Design for Testability (DFT)

In scientific software design, **Design for Testability (DFT)** is derived from a fundamental platform ethos: **underwriting the reliability and insurance of inferential claims**.

### The Underwriting Analogy in Scientific Software
Authors and domain scientists are in the business of *Discovery*—proposing new inferential claims, forward models, and empirical fits. The platform and review teams are in a fundamentally different business: they **underwrite the reliability of those claims**.

Underwriting is the established actuarial practice of assessing proposed risks before accepting them, defining the explicit conditions under which coverage holds, and enforcing proof obligations. Applied to computational science:
- **The proposed risk:** The inferential claim, parameter estimate, or forward model seeking coverage.
- **Coverage:** Reliance—being cited, published, deployed, or used as a trusted basis for downstream pipelines.
- **Underwriting:** The rigorous assessment of whether the computational implementation realizes the stated mathematical and empirical claims.
- **The proof obligation (premium):** The invariant testing, condition budgeting, and error-bound verification that the operator must survive before coverage is granted.
- **The policy:** The stated-and-tested operational boundary—a claim is covered strictly under the conditions it has tested, and never beyond them.

A claim that has not passed its underwriting verification is not refuted; it is simply not yet covered, and remains untrusted.

### The Three Vertices of Underwriting
Every numerical operator in `jaccanon` is underwritten across three independent vertices:
1. **Mathematics:** The equations, operator dualities, and spectral decompositions ($J = S + A$, $\langle S, A \rangle_F = 0$, pushforward/pullback duality).
2. **Software (Implementation):** The code itself—ensuring algorithms expose their conditioning, precision limits, and convergence behavior without silent workarounds.
3. **Data / Numerical Inputs:** Empirical perturbations, condition numbers, and floating-point representations (IEEE 754).

Because silent corruption is ubiquitous in numerical routines—subtractive cancellation, catastrophic truncation, ill-conditioning ($\kappa(J) \gg 10^8$), and pseudo-derivatives at non-differentiable cusps—`jaccanon` enforces an explicit **Diagnostic Contract** across every primitive.

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
