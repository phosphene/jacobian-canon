"""jaccanon.prims.empirical

Canonical 3-point central difference coordinate primitive with full observable
diagnostics, boundary constraints, and condition monitoring.

Lineage: Wilkinson, Richardson, D'Errico.
DFT Pillar: Observability (returns JacobianResult with explicit error bounds).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np

from jaccanon.core.types import JacobianResult


def jacobian_central_3point(
    f: Callable[[np.ndarray], np.ndarray | Sequence[float]],
    x: Sequence[float] | np.ndarray,
    rel_step: float | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
) -> JacobianResult:
    """Compute the Jacobian matrix J = df/dx via 3-point central differences.

    Provides explicit Design for Testability (DFT) diagnostics:
    - Truncation error estimate O(h^2)
    - Subtractive cancellation estimate O(eps_mach / h)
    - Condition number kappa(J) via singular values
    - Pathology detection (boundary collisions, ill-conditioning)

    Args:
        f: Vector-valued function mapping R^n -> R^m.
        x: Base evaluation point of length n.
        rel_step: Relative step size scaling. Default is machine eps^(1/3).
        bounds: Optional tuple of (lower_bounds, upper_bounds) for boundary reflection.

    Returns:
        JacobianResult containing immutable matrix and complete diagnostic manifest.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        msg = f"Base point x must be 1D, got ndim={x_arr.ndim}"
        raise ValueError(msg)

    n = x_arr.size
    eps_mach = float(np.finfo(np.float64).eps)

    # 1. Controllability: step formulation
    # Optimal step for 3-point stencils balancing O(h^2) and O(eps/h) is h ~ eps^(1/3)
    h_base = eps_mach ** (1.0 / 3.0) if rel_step is None else float(rel_step)
    h_vec = h_base * np.maximum(np.abs(x_arr), 1.0)

    # Base primal evaluation
    f_0 = np.asarray(f(x_arr), dtype=np.float64)
    if f_0.ndim != 1:
        msg = f"Function f(x) must return a 1D vector, got ndim={f_0.ndim}"
        raise ValueError(msg)
    m = f_0.size

    j_matrix = np.zeros((m, n), dtype=np.float64)
    eval_count = 1
    pathologies: list[str] = []

    lb_arr = np.asarray(bounds[0], dtype=np.float64) if bounds is not None else None
    ub_arr = np.asarray(bounds[1], dtype=np.float64) if bounds is not None else None

    # 2. Coordinate Loop with Boundary Reflection
    for j in range(n):
        step_j = float(h_vec[j])
        x_plus = x_arr.copy()
        x_minus = x_arr.copy()

        # Reflect step if boundary violation occurs
        if lb_arr is not None and ub_arr is not None:
            if x_arr[j] + step_j > ub_arr[j]:
                step_j = -step_j
                pathologies.append(f"UPPER_BOUND_HIT_COL_{j}")
            elif x_arr[j] - step_j < lb_arr[j]:
                step_j = -step_j
                pathologies.append(f"LOWER_BOUND_HIT_COL_{j}")

        x_plus[j] += step_j
        x_minus[j] -= step_j

        f_plus = np.asarray(f(x_plus), dtype=np.float64)
        f_minus = np.asarray(f(x_minus), dtype=np.float64)
        eval_count += 2

        # Central difference quotient: (f(x+h) - f(x-h)) / (2h)
        j_matrix[:, j] = (f_plus - f_minus) / (2.0 * step_j)

    # 3. Observability Diagnostics: Quantify Truncation and Roundoff
    est_trunc_error = float(np.mean(h_vec**2))
    est_roundoff_error = float(np.mean(eps_mach / h_vec))
    lost_digits = float(np.clip(-np.log10(max(est_roundoff_error, 1e-16)), 0.0, 16.0))

    # Condition number calculation kappa(J)
    s = np.linalg.svd(j_matrix, compute_uv=False)
    if s[-1] > 1e-15:
        cond = float(s[0] / s[-1])
    else:
        cond = float("inf")
        pathologies.append("RANK_DEFICIENT_OR_SINGULAR")

    matrix_tuple = tuple(tuple(float(val) for val in row) for row in j_matrix)

    return JacobianResult(
        matrix=matrix_tuple,
        shape=(m, n),
        method="central-3point",
        genealogy="empirical-wilkinson",
        estimated_truncation_error=est_trunc_error,
        estimated_roundoff_error=est_roundoff_error,
        loss_of_significance_digits=lost_digits,
        condition_number=cond,
        primal_evaluations=eval_count,
        directional_passes=n,
        memory_peak_bytes=j_matrix.nbytes,
        floating_point_precision="float64",
        pathology_flags=tuple(pathologies),
    )
