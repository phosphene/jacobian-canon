"""Tests for jaccanon.prims.empirical verifying DFT contracts."""

import numpy as np
import pytest
from jaccanon.prims.empirical import jacobian_central_3point


def test_linear_function_exact_jacobian():
    # f(x) = A x
    A = np.array([[2.0, -1.0], [3.0, 5.0]])

    def f(x: np.ndarray) -> np.ndarray:
        return A @ x

    x0 = np.array([1.5, -2.0])
    res = jacobian_central_3point(f, x0)

    j_calc = np.array(res.matrix)
    np.testing.assert_allclose(j_calc, A, rtol=1e-8, atol=1e-8)
    assert res.method == "central-3point"
    assert res.genealogy == "empirical-wilkinson"
    assert res.primal_evaluations == 5  # 1 base + 2 per dimension (2 dims -> 4)
    assert res.condition_number >= 1.0


def test_taylor_first_order_consistency():
    # Non-linear test: f(x) = [x0^2 + sin(x1), exp(x0 * x1)]
    def f(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + np.sin(x[1]), np.exp(x[0] * x[1])])

    x0 = np.array([1.0, 0.5])
    res = jacobian_central_3point(f, x0)
    J = np.array(res.matrix)

    # Analytical Jacobian:
    # row 0: [2*x0, cos(x1)] = [2.0, cos(0.5)]
    # row 1: [x1*exp(x0*x1), x0*exp(x0*x1)] = [0.5*exp(0.5), 1.0*exp(0.5)]
    J_exact = np.array([
        [2.0 * x0[0], np.cos(x0[1])],
        [x0[1] * np.exp(x0[0] * x0[1]), x0[0] * np.exp(x0[0] * x0[1])],
    ])

    np.testing.assert_allclose(J, J_exact, rtol=1e-6, atol=1e-6)

    # Verification of Taylor remainder: ||f(x+v) - f(x) - J v|| = O(||v||^2)
    for step in [1e-2, 1e-3, 1e-4]:
        v = np.array([step, -step])
        rem = np.linalg.norm(f(x0 + v) - f(x0) - J @ v)
        # Remainder should scale with step^2
        assert rem <= 10.0 * (step**2)
