# Hypothesis Testing Patterns, Calibration Standards, and Seam Formalization in jacobian-canon

## Executive Mandate

This document codifies the property-based testing and algebraic calibration standards across `jacobian-canon`.

In numerical differentiation and operator decomposition (e.g., Helmholtz symmetric/antisymmetric splitting, Richardson extrapolation, complex-step differentiation, and dual-number manifolds), floating-point error, non-holomorphic boundaries, and ill-conditioned Jacobians must be systematically stressed rather than sampled at arbitrary single points.

---

## 1. Input Geometry and Generator Hygiene

### 1.1 Strict IEEE-754 Finite-Domain Hygiene
* All floating-point inputs in `@given` strategies explicitly declare `allow_nan=False, allow_infinity=False, allow_subnormal=False`.
* Extended-real and pathological branch inputs (e.g. non-holomorphic points where $f(z) = |z|$) are tested in dedicated diagnostic boundary tests rather than unconstrained property sweeps.

### 1.2 Constructive Orthogonality and Manifold Projection
* Manifolds such as symmetric positive-definite cones and orthogonal Lie algebras $\mathfrak{so}(n)$ are parameterized constructively via matrix projection:
  - Symmetric part: $S = \frac{1}{2}(M + M^T)$
  - Antisymmetric part: $A = \frac{1}{2}(M - M^T)$
* Rather than rejection-filtering random matrices, constructive splitting guarantees that $100\%$ of generated instances satisfy exact reconstruction:
  $$M \equiv S + A$$
  and Frobenius orthogonality:
  $$\langle S, A \rangle_F = \text{tr}(S^T A) = 0$$

---

## 2. Oracles and Variational Defect Hunting

### 2.1 Cross-Method Consistency Sandwiches
Complex-step differentiation ($O(h^2)$ without step cancellation down to machine precision) serves as the primary analytical reference oracle against which higher-order Richardson extrapolation tableaus are calibrated.

### 2.2 Variational Defect Hunting (`hypothesis.target()`)
Residuals between analytical baselines and numerical tableaus are exposed as continuous non-negative functionals:
```python
defect = float(np.linalg.norm(J_richardson - J_complex_step, ord=np.inf))
target(defect, label="richardson_vs_complex_step_defect")
assert defect <= 1e-10
```
This forces Hypothesis to actively optimize toward coordinates where nonlinear curvature and higher-order Taylor truncation error are maximized.
