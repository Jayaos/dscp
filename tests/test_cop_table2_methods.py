"""Regression checks for COP's Table 2 baseline implementations.

Hand-calculated trajectories lock the authors' warmup and indexing conventions.
The vendored source is MIT licensed; see baselines/cop_table2/UPSTREAM_LICENSE.
"""

import numpy as np
import pytest

from baselines.cop_table2.upstream_methods import COP, LQT, aci


def test_aci_warmup_uses_linear_quantile_then_higher_and_strict_boundary():
    scores = np.arange(130, dtype=float) - 60.0
    result = aci(scores, .05, .1, window_length=10_000_000, T_burnin=100, ahead=1)
    q = result["q"]
    assert np.all(np.isposinf(q[:21]))
    assert q[21] == -41.0
    assert q[100] == pytest.approx(34.05)
    assert q[101] == 35.0
    assert np.all(result["alpha"][:102] == .05)
    assert result["alpha"][102] == pytest.approx(-.045)
    assert np.isposinf(q[102])
    assert result["alpha"][103] == pytest.approx(-.04)


def test_aci_burnin_errors_do_not_adapt_state_and_negative_thresholds_survive():
    scores = np.full(130, -2.0)
    scores[100:102] = 100.0
    result = aci(scores, .05, .1, window_length=10_000_000, T_burnin=100, ahead=1)
    assert result["q"][100] == -2.0
    assert result["q"][101] == -2.0
    assert result["alpha"][101] == .05
    assert result["alpha"][102] == pytest.approx(-.045)
    assert np.isposinf(result["q"][102])


def test_aci_upper_alpha_clips_quantile_level_without_clipping_state():
    scores = np.full(108, -1.0)
    result = aci(scores, .05, 10.0, window_length=10_000_000, T_burnin=100, ahead=1)
    assert result["alpha"][103] > 1.0
    assert np.all(result["q"][101:] == -1.0)


def test_lqt_signed_hand_calculated_trajectory_and_equality_is_covered():
    scores = np.array([-2., 1., 3., -1., 4., -3.])
    result = LQT(scores, .25, .1, ahead=1, T_burnin=4, p_order=2, bias=5.)
    np.testing.assert_allclose(result["q"], [0., 0., 3., 6.1, 4.175, 3.875], rtol=0, atol=1e-14)


def test_lqt_preserves_authors_initialization_from_future_burnin_scores():
    scores = np.array([-2., 1., 3., -1., 4., -3.])
    altered = scores.copy()
    altered[3] = 100.0
    args = dict(alpha=.25, lr=.1, ahead=1, T_burnin=4, p_order=2, bias=5.)
    assert LQT(scores, **args)["q"][2] == 3.0
    assert LQT(altered, **args)["q"][2] == 54.5


def test_lqt_proportional_flag_is_unused_in_the_published_code():
    scores = np.array([-2., 1., 3., -1., 4., -3.])
    args = dict(alpha=.25, lr=.1, ahead=1, T_burnin=4, p_order=2, bias=5.)
    np.testing.assert_array_equal(
        LQT(scores, proportional_lr=True, **args)["q"],
        LQT(scores, proportional_lr=False, **args)["q"],
    )


def test_cop_ecdf_starts_at_burnin_uses_past_scores_and_primary_quantile():
    scores = np.array([2., 4., 8., -1., 3., 5.])
    result = COP(scores, .25, 1., T_burnin=2, ahead=1, proportional_lr=False)
    # At t=2 the CDF is evaluated at 2.25 on [2, 4], giving refinement .125.
    # At t=3 the primary quantile is 2.25, while the predicted quantile is 2.375.
    np.testing.assert_array_equal(result["q"], [0., .75, 1.5, 2.375, 2.375, 2.875])


def test_cop_zero_range_gives_zero_learning_rate_without_fallback():
    scores = np.array([5., 5., 8., 9., 0.])
    result = COP(scores, .1, .5, T_burnin=100, ahead=1)
    np.testing.assert_allclose(result["q"], [0., .45, .45, .45, 1.8], rtol=0, atol=1e-15)


def test_cop_equality_is_covered_and_negative_signed_quantiles_survive():
    scores = np.array([0., -1., -1., -1.])
    result = COP(scores, .25, 1., T_burnin=3, ahead=1, proportional_lr=False)
    np.testing.assert_array_equal(result["q"], [0., -.25, -.5, -.75])


def test_cop_does_not_reset_global_random_state():
    rng = np.random.RandomState(123)
    np.random.set_state(rng.get_state())
    expected = rng.random_sample(4)
    COP(np.array([2., 4., 8., -1.]), .25, 1., T_burnin=2, ahead=1)
    np.testing.assert_array_equal(np.random.random_sample(4), expected)
