"""Tests for jaccanon.reproductions.claims_eval."""

import pytest
from jaccanon.reproductions.claims_eval import (
    evaluate_finite_difference_precision_bounds,
    evaluate_complex_step_claim,
    evaluate_torch_func_ad_claims,
)


def test_finite_difference_precision_claims():
    evals = evaluate_finite_difference_precision_bounds()
    assert len(evals) == 2
    for ev in evals:
        assert ev.verified, f"Claim failed verification: {ev}"


def test_complex_step_claims_and_breakdowns():
    evals = evaluate_complex_step_claim()
    assert len(evals) == 2
    for ev in evals:
        assert ev.verified, f"Claim failed verification: {ev}"


def test_torch_func_ad_claims_and_cusp_behavior():
    evals = evaluate_torch_func_ad_claims()
    assert len(evals) == 2
    for ev in evals:
        assert ev.verified, f"Claim failed verification: {ev}"
