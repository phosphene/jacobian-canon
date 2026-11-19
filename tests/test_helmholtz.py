"""Tests for Helmholtz S+A decomposition in jaccanon.decomposition."""

import numpy as np
from hypothesis import event, given, settings, target
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


@settings(max_examples=100)
@given(
    matrix_data=st.lists(
        st.lists(
            st.floats(
                min_value=-1e4,
                max_value=1e4,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=3,
            max_size=3,
        ),
        min_size=3,
        max_size=3,
    )
)
def test_hypothesis_helmholtz_orthogonality_and_reconstruction(matrix_data):
    """Verify exact algebraic reconstruction M = S + A and Frobenius orthogonality <S, A>_F = 0.
    
    Uses variational target() to actively climb toward the worst-case numerical cancellation
    and tracks condition-number/norm decades via event().
    """
    M = np.array(matrix_data, dtype=np.float64)
    res = decompose_matrix_sa(M)

    S = np.array(res.symmetric_part)
    A = np.array(res.antisymmetric_part)

    # 1. Exact reconstruction invariant: M == S + A
    reconstruction_defect = float(np.max(np.abs((S + A) - M)))
    target(reconstruction_defect, label="helmholtz_reconstruction_defect")
    np.testing.assert_allclose(S + A, M, rtol=1e-12, atol=1e-12)

    # 2. Frobenius orthogonality invariant: <S, A>_F = tr(S^T A) = 0
    inner_product = float(np.sum(S * A))
    orthogonality_defect = abs(inner_product)
    norm_scale = res.frobenius_norm_s * res.frobenius_norm_a + 1.0
    relative_ortho_defect = orthogonality_defect / norm_scale

    target(relative_ortho_defect, label="helmholtz_relative_orthogonality_defect")
    assert orthogonality_defect < 1e-8 * norm_scale
