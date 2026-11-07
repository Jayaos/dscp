"""Scientific fidelity tests for the deterministic published COP path."""

from copy import deepcopy

import numpy as np
import pytest

from baselines.cop.model import COP, MODEL_DEFAULTS, predict_thresholds, validate_model


def reference_cop(scores, alpha, lr, window, proportional=True, scale=.5, initial=0.):
    """Array-form reference retaining the published indexing and update order.

    Adapted from creator-xi/Conformal-Optimistic-Prediction, core/methods.py
    at a41c0dcadc10979d8521ab9b5b8f3d460807d883.
    MIT License, Copyright (c) 2025 Junxi Wu; full notice in
    baselines/cop/UPSTREAM_LICENSE.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    q = np.zeros(n)
    primary = np.zeros(n)
    optimistic = np.zeros(n)
    q[0] = primary[0] = initial
    for t in range(n):
        past = scores[max(0, t-window):t]
        rate = lr * (past.max() - past.min()) if proportional and t > 0 else lr
        gradient = alpha if q[t] >= scores[t] else -(1-alpha)
        if t >= window:
            optimistic[t] = np.mean(scores[t-window:t] <= primary[t] - rate * gradient) - (1-alpha)
        if t < n - 1:
            primary[t+1] = primary[t] - rate * gradient
            integrator = -scale * optimistic[t]
            q[t+1] = primary[t+1] + rate * integrator
    return q, primary, optimistic


@pytest.mark.parametrize("proportional", [True, False])
@pytest.mark.parametrize("window", [1, 3, 100])
@pytest.mark.parametrize("alpha", [.05, .2])
@pytest.mark.parametrize("score_type", ["signed", "absolute"])
def test_matches_published_array_recurrence(proportional, window, alpha, score_type):
    time = np.arange(125)
    scores = 2 * np.sin(time * .7) + .01 * time - .4
    if score_type == "absolute":
        scores = np.abs(scores)
    settings = {"learning_rate": .5, "window_size": window, "proportional_lr": proportional}
    result = predict_thresholds(scores, alpha, settings)
    q, primary, optimistic = reference_cop(scores, alpha, .5, window, proportional)
    np.testing.assert_array_equal(result["thresholds"], q)
    np.testing.assert_array_equal(result["primary_thresholds"], primary)
    np.testing.assert_array_equal(result["optimistic_gradients"], optimistic)
    np.testing.assert_array_equal(result["errors"], (scores > q).astype(int))


def test_cdf_uses_stale_window_and_starts_at_window_boundary():
    model = COP(alpha=.1, learning_rate=1., window_size=3, proportional_lr=False)
    traces = [model.observe(score) for score in [2., 4., 8., 100.]]
    assert [row["cdf_active"] for row in traces] == [False, False, False, True]
    assert [row["history_size_used"] for row in traces] == [0, 1, 2, 3]
    last = traces[-1]
    assert last["primary_next"] == 3.6
    assert last["optimistic_gradient"] == np.mean(np.array([2., 4., 8.]) <= 3.6) - .9
    assert last["threshold_next"] == last["primary_next"] + last["refinement"]
    assert model.state_dict()["history"] == [4., 8., 100.]


def test_proportional_rate_preserves_zero_range_without_fallback():
    result = predict_thresholds([5., 5., 8., 9.], .1, {
        "learning_rate": .5, "window_size": 100,
    })
    np.testing.assert_array_equal(result["learning_rates"], [.5, 0., 0., 1.5])
    assert result["thresholds"][1] == result["thresholds"][2] == result["thresholds"][3]


def test_equality_counts_as_covered_and_negative_signed_thresholds_survive():
    estimator = COP(alpha=.1, learning_rate=1., proportional_lr=False, window_size=3)
    diagnostic = estimator.observe(0.)
    assert diagnostic["error"] == 0
    assert diagnostic["gradient"] == .1
    assert estimator.predict() == -.1
    result = predict_thresholds([-1.] * 4, .1, {
        "learning_rate": 1., "proportional_lr": False, "window_size": 3,
    })
    assert np.all(result["thresholds"][1:] < 0)
    np.testing.assert_array_equal(result["errors"], np.zeros(4, dtype=int))


def test_calibration_continuation_matches_one_uninterrupted_stream():
    scores = np.array([2., -1., 3., -2., .5, 4., -3., 1.])
    settings = {"learning_rate": .1, "window_size": 3, "scale": .5, "proportional_lr": True}
    full = predict_thresholds(scores, .05, settings)
    estimator = COP(alpha=.05, **settings)
    for value in scores[:5]:
        estimator.observe(float(value))
    thresholds = []
    for value in scores[5:]:
        thresholds.append(estimator.predict())
        estimator.observe(float(value))
    np.testing.assert_array_equal(thresholds, full["thresholds"][5:])
    assert estimator.state_dict() == full["final_state"]


def test_current_and_future_scores_cannot_change_issued_threshold():
    scores = np.array([1., -1., 3., 2., 4., -2., 5.])
    changed = scores.copy()
    changed[4:] = [100., -50., 200.]
    for proportional in (True, False):
        settings = {"window_size": 2, "proportional_lr": proportional}
        original = predict_thresholds(scores, .1, settings)
        altered = predict_thresholds(changed, .1, settings)
        np.testing.assert_array_equal(original["thresholds"][:5], altered["thresholds"][:5])
        prefix = predict_thresholds(scores[:5], .1, settings)
        np.testing.assert_array_equal(original["thresholds"][:5], prefix["thresholds"])


def test_independent_instances_and_detached_state_snapshots():
    first = COP(window_size=2)
    second = COP(window_size=2)
    first.observe(10.)
    assert second.predict() == 0
    assert second.seen == 0
    snapshot = first.state_dict()
    snapshot["history"].append(999.)
    snapshot["threshold"] = -999
    assert first.history_size == 1
    assert first.predict() == pytest.approx(.09)


def test_no_global_random_state_mutation():
    before = np.random.get_state()
    predict_thresholds([1., -2., 3., 4.], .1, {"window_size": 2})
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_configurable_initial_quantile_scale_and_zero_rate():
    settings = {"initial_quantile": 2., "scale": .25, "window_size": 2,
                "learning_rate": .1, "proportional_lr": False}
    scores = [-1., 3., 5., -2., 4.]
    output = predict_thresholds(scores, .1, settings)
    expected, _, _ = reference_cop(scores, .1, .1, 2, False, .25, 2.)
    np.testing.assert_array_equal(output["thresholds"], expected)
    output = predict_thresholds(scores, .1, {**settings, "learning_rate": 0.})
    np.testing.assert_array_equal(output["thresholds"], np.full(len(scores), 2.))


def test_scale_zero_is_unmodified_gradient_descent():
    settings = {"learning_rate": .1, "scale": 0., "window_size": 2, "proportional_lr": False}
    result = predict_thresholds([1., -1., 3., -4.], .1, settings)
    np.testing.assert_array_equal(result["thresholds"], result["primary_thresholds"])
    np.testing.assert_array_equal(result["refinements"], np.zeros(4))


def test_defaults_and_runner_extras_are_preserved_without_mutating_input():
    original = {"window_size": 4, "target_quantiles": [[.05, .95]]}
    before = deepcopy(original)
    settings = validate_model(original)
    assert settings["scale"] == .5
    assert settings["proportional_lr"] is True
    assert settings["cdf"] == "ecdf"
    settings["target_quantiles"][0][0] = .1
    assert original == before
    assert MODEL_DEFAULTS["window_size"] == 100


@pytest.mark.parametrize("key,value", [
    ("learning_rate", -1), ("learning_rate", float("nan")), ("learning_rate", True),
    ("window_size", 0), ("window_size", 1.5), ("window_size", True),
    ("scale", -1), ("scale", float("inf")), ("proportional_lr", 1),
    ("cdf", "kde"), ("initial_quantile", float("nan")),
])
def test_invalid_model_settings(key, value):
    with pytest.raises(ValueError):
        validate_model({key: value})


@pytest.mark.parametrize("alpha", [0, 1, -.1, float("nan"), True])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        COP(alpha=alpha)


@pytest.mark.parametrize("scores", [[], [[1.]], [float("nan")], [float("inf")], ["one"], [True]])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):
        predict_thresholds(scores, .1, {})


def test_rate_overflow_is_explicit_and_does_not_commit_state():
    estimator = COP(learning_rate=1e308, window_size=2, proportional_lr=True)
    estimator.observe(-1.)
    estimator.observe(1.)
    before = estimator.state_dict()
    with pytest.raises(RuntimeError, match="learning rate overflowed"):
        estimator.observe(0.)
    assert before == estimator.state_dict()


def test_primary_overflow_is_explicit():
    estimator = COP(learning_rate=1e308, proportional_lr=False, initial_quantile=1e308)
    with pytest.raises(RuntimeError, match="primary threshold overflowed"):
        estimator.observe(1.7e308)
    assert estimator.seen == 0


def test_refinement_overflow_is_explicit():
    estimator = COP(learning_rate=1e308, proportional_lr=False, window_size=1, scale=100.)
    estimator.observe(-1.)
    with pytest.raises(RuntimeError, match="optimistic threshold overflowed"):
        estimator.observe(-1.)
