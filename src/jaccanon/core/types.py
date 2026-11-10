"""jaccanon.core.types

Immutable data models and result containers adhering to Phosphene Python
engineering standards:
- Base class: `phosphene.core.models.InvariantStatic` (frozen=True, extra='forbid')
- Observable diagnostics and error bounds upfront
- Strict typing and Any-budget compliance
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field

from phosphene.core.models import InvariantStatic


class JacobianResult(InvariantStatic):
    """Standardized Public Science Result Container exposing all numerical

    tradeoffs, error bounds, conditioning, and resource expenditures upfront.
    """

    # Materialized Jacobian matrix represented as an immutable tuple of rows
    matrix: tuple[tuple[float, ...], ...] = Field(
        description="Materialized Jacobian matrix J of shape (m, n)"
    )
    shape: tuple[int, int] = Field(description="Dimensions (m, n) of the Jacobian matrix")
    method: str = Field(description="Identifier of numerical method used (e.g., 'central-3point')")
    genealogy: str = Field(description="Historical lineage (e.g., 'empirical-wilkinson', 'analytic-ad')")

    # Error Bounds and Conditioning (Observability)
    estimated_truncation_error: float = Field(
        ge=0.0, description="Estimated asymptotic truncation error O(h^p)"
    )
    estimated_roundoff_error: float = Field(
        ge=0.0, description="Estimated floating-point roundoff cancellation error O(eps_mach / h)"
    )
    loss_of_significance_digits: float = Field(
        ge=0.0, le=16.0, description="Estimated decimal digits lost to subtractive cancellation"
    )
    condition_number: float = Field(
        ge=1.0, description="SVD-based matrix condition number kappa(J)"
    )

    # Computational Complexity Budget
    primal_evaluations: int = Field(ge=1, description="Total raw evaluations of primal function f(x)")
    directional_passes: int = Field(ge=0, description="Number of directional pushforward or pullback passes")
    memory_peak_bytes: int = Field(ge=0, description="Peak memory consumed during computation")
    floating_point_precision: Literal["float32", "float64", "mpmath-128"] = Field(
        default="float64", description="Floating-point precision representation"
    )

    # Diagnostic Warnings
    pathology_flags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Active numerical pathology flags (e.g., 'CUSP_DETECTED', 'ILL_CONDITIONED')",
    )


class SADecompositionResult(InvariantStatic):
    """Result of Helmholtz symmetric/antisymmetric matrix decomposition (M = S + A)."""

    symmetric_part: tuple[tuple[float, ...], ...] = Field(
        description="Symmetric component S = (M + M^T) / 2 representing conservative potential gradient"
    )
    antisymmetric_part: tuple[tuple[float, ...], ...] = Field(
        description="Antisymmetric component A = (M - M^T) / 2 representing non-conservative rotational curl"
    )
    frobenius_norm_s: float = Field(ge=0.0, description="Frobenius norm ||S||_F")
    frobenius_norm_a: float = Field(ge=0.0, description="Frobenius norm ||A||_F")
    asymmetry_ratio: float = Field(
        ge=0.0, description="Ratio rho = ||A||_F / ||S||_F; 0 indicates purely conservative field"
    )
    regime: Literal["conservative", "non_conservative"] = Field(
        description="Classified dynamic regime based on symmetrization threshold"
    )
    recommended_solver: Literal["gradient_descent", "extragradient_projection"] = Field(
        description="Automated solver routing decision"
    )


class EquilibriumResult(InvariantStatic):
    """Result of stationary point solver execution over dynamical vector fields."""

    solution: tuple[float, ...] = Field(description="Equilibrium state vector z*")
    residual: float = Field(ge=0.0, description="Operator norm ||f(z*)|| at termination")
    iterations: int = Field(ge=0, description="Total iterations executed")
    converged: bool = Field(description="True if convergence tolerance was satisfied")
    solver_used: str = Field(description="Identifier of solver algorithm executed")
