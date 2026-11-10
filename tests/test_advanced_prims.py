"""Tests for jaccanon.prims.complex_step and jaccanon.prims.richardson."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jaccanon.prims.complex_step import jacobian_complex_step
from jaccanon.prims.richardson import jacobian_richardson


def test_complex_step_smooth_nonlinear_analytic():
    # f(x) = [exp(x0) * cos(x1), sin(x0 * x1)]
    def f(x):
        return np.array([
            np.exp(x[0]) * np.cos(x[1]),
            np.sin(x[0] * x[1])
        ])

    x0 = np.array([1.2, 0.7])
    res = jacobian_complex_step(f, x0)

    # Exact analytical Jacobian:
    # row 0: [exp(x0)*cos(x1), -exp(x0)*sin(x1)]
    # row 1: [x1*cos(x0*x1),   x0*cos(x0*x1)]
    J_exact = np.array([
        [np.exp(x0[0]) * np.cos(x0[1]), -np.exp(x0[0]) * np.sin(x0[1])],
        [x0[1] * np.cos(x0[0] * x0[1]), x0[0] * np.cos(x0[0] * x0[1])],
    ])

    J_calc = np.array(res.matrix)
    # Complex-step achieves machine precision (~1e-15 or better)
    np.testing.assert_allclose(J_calc, J_exact, rtol=1e-14, atol=1e-14)
    assert res.method == "complex-step"
    assert res.genealogy == "analytic-lyness-moler"
    assert res.loss_of_significance_digits == 0.0


def test_complex_step_non_holomorphic_pathology_flag():
    # f(x) = [abs(x0), exp(x1)]
    # abs(z) is non-holomorphic; np.abs(1.0 + 1j*h) has imag part 0
    def f_non_holo(x):
        return np.array([np.abs(x[0]), np.exp(x[1])])

    x0 = np.array([1.0, 0.5])
    res = jacobian_complex_step(f_non_holo, x0)
    # Should flag NON_HOLOMORPHIC_ZERO_IMAG_COL_0
    assert any("NON_HOLOMORPHIC" in flag for flag in res.pathology_flags)


def test_richardson_high_precision_extrapolation():
    # f(x) = [x0^3 + 2*x1^2, exp(x0 + x1)]
    def f(x):
        return np.array([
            x[0]**3 + 2.0 * (x[1]**2),
            np.exp(x[0] + x[1])
        ])

    x0 = np.array([1.5, -0.8])
    res = jacobian_richardson(f, x0, tableau_depth=4)

    # Exact Jacobian:
    # row 0: [3*x0^2, 4*x1] = [3*(1.5)^2, 4*(-0.8)] = [6.75, -3.2]
    # row 1: [exp(x0+x1), exp(x0+x1)]
    exp_val = np.exp(x0[0] + x0[1])
    J_exact = np.array([
        [6.75, -3.2],
        [exp_val, exp_val],
    ])

    J_calc = np.array(res.matrix)
    # Richardson eliminates O(h^2), O(h^4), O(h^6), achieving precision down to 1e-12
    np.testing.assert_allclose(J_calc, J_exact, rtol=1e-11, atol=1e-11)
    assert res.method == "richardson-tableau"
    assert res.genealogy == "extrapolation-richardson-derrico"
    assert res.estimated_truncation_error < 1e-10


@settings(max_examples=30)
@given(
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_complex_step_vs_richardson_consistency(x0_val, x1_val):
    # Smooth analytic function: f(x) = [sin(x0) + cos(x1), exp(0.5*x0)]
    def f(x):
        return np.array([
            np.sin(x[0]) + np.cos(x[1]),
            np.exp(0.5 * x[0])
        ])

    x_pt = np.array([x0_val, x1_val])
    res_cs = jacobian_complex_step(f, x_pt)
    res_rich = jacobian_richardson(f, x_pt, tableau_depth=4)

    J_cs = np.array(res_cs.matrix)
    J_rich = np.array(res_rich.matrix)

    # Both independent methods must agree to within 1e-10 across continuous parameter space
    np.testing.assert_allclose(J_cs, J_rich, rtol=1e-9, atol=1e-9)
