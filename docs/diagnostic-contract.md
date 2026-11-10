---
title: The Diagnostic Contract and Design for Testability (DFT)
author: Ed Phil & Flow (@feelingflowingbot)
status: living
created: 2026-11-10
---

# The Diagnostic Contract and Design for Testability (DFT)

In safety-critical hardware engineering and VLSI, **Design for Testability (DFT)** establishes that testability is not an afterthought; internal states, error accumulation, and structural invariants must be observable and controllable through the interface itself.

In numerical software, silent corruption is ubiquitous:
- Finite-difference quotients lose half their precision to subtractive cancellation ($f(x+h) - f(x)$ when $h$ is small).
- Truncation errors grow catastrophically when step sizes are chosen too aggressively.
- Ill-conditioned operators ($\kappa(J) \gg 10^8$) invert numerical noise into physical signal.
- Non-differentiable points and branch cuts produce pseudo-derivatives that mislead optimization algorithms.

`jaccanon` enforces a strict **Diagnostic Contract** across every primitive.

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
