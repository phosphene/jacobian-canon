"""Tests for Helmholtz S+A decomposition in jaccanon.decomposition."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from jaccanon.decomposition.helmholtz import decompose_matrix_sa


def test_purely_symmetric_matrix():
    M = np.array([[2.0, 1.0], [1.0, 3.0]])
    res = decompose_matrix_sa(M)
    assert res.regime == "conservative"
    assert res.recommended_solver == "gradient_descent"
    assert res.frobenius_norm_a == 0.0
    assert res.asymmetry_ratio == 0.0


def test_purely_antisymmetric_matrix():
    M = np.array([[0.0, 2.0], [-2.0, 0.0]])
    res = decompose_matrix_sa(M)
    assert res.regime == "non_conservative"
    assert res.recommended_solver == "extragradient_projection"
    assert res.frobenius_norm_s == 0.0
    assert res.asymmetry_ratio == float("inf")


@settings(max_examples=50)
@given(
    st.lists(
        st.lists(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False), min_size=3, max_size=3),
        min_size=3,
        max_size=3,
    )
)
def test_hypothesis_helmholtz_orthogonality_and_reconstruction(matrix_data):
    M = np.array(matrix_data, dtype=np.float64)
    res = decompose_matrix_sa(M)

    S = np.array(res.symmetric_part)
    A = np.array(res.antisymmetric_part)

    # 1. Exact reconstruction invariant: M == S + A
    np.testing.assert_allclose(S + A, M, rtol=1e-12, atol=1e-12)

    # 2. Frobenius orthogonality invariant: <S, A>_F = tr(S^T A) = 0
    inner_product = float(np.sum(S * A))
    assert abs(inner_product) < 1e-8 * (res.frobenius_norm_s * res.frobenius_norm_a + 1.0)
