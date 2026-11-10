"""jaccanon.decomposition.helmholtz

Helmholtz S+A matrix decomposition and automated solver routing.

Axiomatic reduction:
For any square matrix M in R^{n x n}, decomposes into:
    M = S + A
where:
    S = 0.5 * (M + M^T)  [Symmetric: conservative potential gradient]
    A = 0.5 * (M - M^T)  [Antisymmetric: non-conservative rotational curl]

Frobenius orthogonality:
    <S, A>_F = tr(S^T A) = 0

Symmetrization threshold criterion:
    rho = ||A||_F / ||S||_F < eps_sym (default: 0.05)
    rho < eps_sym  -> Conservative regime (gradient descent)
    rho >= eps_sym -> Non-conservative regime (Korpelevich extragradient)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
import numpy as np

from jaccanon.core.types import SADecompositionResult

DEFAULT_SYMMETRIZATION_EPSILON: float = 0.05


def decompose_matrix_sa(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    epsilon_sym: float = DEFAULT_SYMMETRIZATION_EPSILON,
) -> SADecompositionResult:
    """Perform Helmholtz S+A decomposition on a Jacobian or transmission matrix.

    Args:
        matrix: Square 2D matrix of shape (n, n).
        epsilon_sym: Symmetrization threshold ratio ||A||_F / ||S||_F below which
            the field is classified as conservative (default 0.05).

    Returns:
        SADecompositionResult immutable record with norms and solver routing.

    Raises:
        ValueError: If matrix is not 2D or not square.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        msg = f"Expected square 2D matrix, got shape {arr.shape}"
        raise ValueError(msg)

    s_arr = 0.5 * (arr + arr.T)
    a_arr = 0.5 * (arr - arr.T)

    norm_s = float(np.linalg.norm(s_arr, ord="fro"))
    norm_a = float(np.linalg.norm(a_arr, ord="fro"))

    if norm_s == 0.0:
        ratio = 0.0 if norm_a == 0.0 else float("inf")
    else:
        ratio = float(norm_a / norm_s)

    is_conservative = ratio < epsilon_sym
    regime: Literal["conservative", "non_conservative"] = (
        "conservative" if is_conservative else "non_conservative"
    )
    recommended_solver: Literal["gradient_descent", "extragradient_projection"] = (
        "gradient_descent" if is_conservative else "extragradient_projection"
    )

    s_tuple = tuple(tuple(float(x) for x in row) for row in s_arr)
    a_tuple = tuple(tuple(float(x) for x in row) for row in a_arr)

    return SADecompositionResult(
        symmetric_part=s_tuple,
        antisymmetric_part=a_tuple,
        frobenius_norm_s=norm_s,
        frobenius_norm_a=norm_a,
        asymmetry_ratio=ratio,
        regime=regime,
        recommended_solver=recommended_solver,
    )
