"""Numerical and causal fidelity checks for the published online algorithms."""

import numpy as np
import pytest

from baselines.conformalopt.model import (
    compute_ar_scorecasts, predict_thresholds, validate_model,
)


def _model(**kwargs):
    return validate_model(kwargs)


def _original_lqt_equations(scores, alpha, model):
    """Array-form recurrence, independently derived from the experiment equations."""
    p = model["p_order"]
    values = np.asarray(scores, dtype=float)
    rows = len(values)
    weights = np.zeros((rows + 1, p + 1))
    weights[p] = np.repeat(1 / p if p else model["initial_quantile"], p + 1)
    features = np.zeros((rows, p + 1))
    for row in range(p, rows):
        features[row, :-1] = values[row - p:row]
        features[row, -1] = model["bias"]
    predictions = []
    for row in range(rows):
        prediction = np.inner(weights[row], features[row])
        predictions.append(prediction)
        if row >= p:
            rate = model["learning_rate"]
            if model["learning_rate_schedule"] == "decaying":
                rate /= (row + 1) ** model["decay_exponent"]
            gradient = (float(prediction < values[row]) - alpha) * features[row]
            weights[row + 1] = weights[row] + rate * gradient
    return np.array(predictions)


def _original_ar_equations(scores, alpha, model):
    """Direct fits over indexed past rows, followed by a scalar recurrence."""
    cp = pytest.importorskip("cvxpy")
    scores = np.asarray(scores, dtype=float)
    p = model["p_order"]
    quantile = 1 - alpha
    result = np.zeros(len(scores))
    correction = 0.0
    for t in range(p, len(scores)):
        # The initial p-1 row is zero; all later rows contain p lags + 1.
        indices = range(max(p - 1, t - model["ar_window_size"] - 1), t)
        features = np.array([
            [0.0] * (p + 1) if k == p - 1 else [*scores[k - p:k], 1.0]
            for k in indices
        ])
        observed = np.array([scores[k] if k >= 0 else 0 for k in indices])
        beta = cp.Variable(p + 1)
        residuals = observed - features @ beta
        loss = cp.maximum(quantile * residuals, (quantile - 1) * residuals)
        cp.Problem(cp.Minimize(cp.sum(loss))).solve(solver="CLARABEL")
        prediction_features = np.array([*scores[t - p:t], 1.0])
        result[t] = prediction_features @ beta.value + correction
        rate = model["learning_rate"]
        if model["learning_rate_schedule"] == "decaying":
            rate /= (t + 1) ** model["decay_exponent"]
        correction += rate * (float(scores[t] > result[t]) - alpha)
    return result


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("schedule", ["fixed", "decaying"])
def test_lqt_matches_experiment_recurrence(order, schedule):
    scores = np.array([0.0, 0.3, 1.7, 0.2, 0.5, 3.0, 1.0, 0.9])
    model = _model(p_order=order, learning_rate=.7, bias=5,
                   learning_rate_schedule=schedule)
    actual = predict_thresholds(scores, .1, model)
    expected = _original_lqt_equations(scores, .1, model)
    np.testing.assert_allclose(actual["thresholds"], expected, atol=1e-12)
    np.testing.assert_array_equal(actual["errors"], scores > expected)


def test_lqt_warmup_initialization_and_equality_subgradient():
    # p=2: theta=(.5,.5,.5), first usable phi=(2,4,2), hence threshold=4.
    model = _model(p_order=2, bias=2, learning_rate=1)
    result = predict_thresholds([2., 4., 4., 1.], .1, model)
    np.testing.assert_allclose(result["thresholds"], [0., 0., 4., 2.2])
    # Covered equality uses -alpha, not 1-alpha: theta becomes(.3,.1,.3).
    np.testing.assert_allclose(result["final_theta"], [.3, .1, .3])
    np.testing.assert_array_equal(result["errors"], [1, 1, 0, 0])


def test_scalar_initial_quantile_bias_and_unclipped_negative_threshold():
    result = predict_thresholds([0., 0., 0.], .1,
                                _model(p_order=0, initial_quantile=0,
                                       bias=1, learning_rate=100))
    np.testing.assert_allclose(result["thresholds"], [0., -10., 80.])
    np.testing.assert_array_equal(result["errors"], [0, 1, 0])
    biased = predict_thresholds([4., 0.], .1,
                                _model(p_order=0, initial_quantile=2,
                                       bias=2, learning_rate=1))
    np.testing.assert_allclose(biased["thresholds"], [4., 3.6])


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("schedule", ["fixed", "decaying"])
def test_sqt_ar_matches_original_fits_and_scalar_correction(order, schedule):
    pytest.importorskip("cvxpy")
    values = [.2, .7, 1.4, .3, .9, 2.0, .1, .6]
    model = _model(method="sqt_ar", p_order=order, ar_window_size=2,
                   learning_rate=.7, learning_rate_schedule=schedule)
    actual = predict_thresholds(values, .1, model)
    expected = _original_ar_equations(values, .1, model)
    np.testing.assert_allclose(actual["thresholds"], expected, atol=1e-7, rtol=1e-7)
    assert set(actual["solver_status"][order:]) <= {"optimal", "optimal_inaccurate"}
    np.testing.assert_array_equal(actual["ar_training_rows"][order:],
                                  np.minimum(np.arange(1, len(values) - order + 1), 3))


@pytest.mark.parametrize("method", ["lqt", "sqt_ar"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_predictions_use_only_strictly_past_scores(method, order):
    if method == "sqt_ar":
        pytest.importorskip("cvxpy")
    model = _model(method=method, p_order=order, learning_rate=.15)
    prefix = np.array([.2, .8, .3, .7, .1])
    a = predict_thresholds(np.r_[prefix, 1., 3.], .1, model)["thresholds"]
    b = predict_thresholds(np.r_[prefix, 300., 1000.], .1, model)["thresholds"]
    # Even prediction at the first changed outcome must be identical.
    np.testing.assert_allclose(a[:len(prefix) + 1], b[:len(prefix) + 1], atol=1e-10)
    c = predict_thresholds(prefix, .1, model)["thresholds"]
    np.testing.assert_allclose(a[:len(prefix)], c, atol=1e-10)


def test_ar_window_201_rows_matches_original_off_by_one():
    pytest.importorskip("cvxpy")
    model = _model(method="sqt_ar", p_order=1, ar_window_size=200)
    values = np.random.default_rng(4).uniform(.1, 2, 204)
    result = compute_ar_scorecasts(values, .1, model)
    assert result["ar_training_rows"][0] == 0
    np.testing.assert_array_equal(result["ar_training_rows"][1:202], np.arange(1, 202))
    np.testing.assert_array_equal(result["ar_training_rows"][202:], [201, 201])


def test_ar_cache_reused_across_learning_rates_and_validated(monkeypatch):
    pytest.importorskip("cvxpy")
    values = np.array([.2, .4, .7, .3])
    model = _model(method="sqt_ar", p_order=1)
    cache = compute_ar_scorecasts(values, .1, model)
    def fail(*args, **kwargs):
        raise AssertionError("A cached scorecast should not solve another regression")
    monkeypatch.setattr("baselines.conformalopt.model.compute_ar_scorecasts", fail)
    a = predict_thresholds(values, .1, model, scorecasts=cache)
    b = predict_thresholds(values, .1, {**model, "learning_rate": 2}, scorecasts=cache)
    np.testing.assert_array_equal(a["scorecasts"], b["scorecasts"])
    assert not np.array_equal(a["thresholds"], b["thresholds"])
    with pytest.raises(ValueError, match="different score sequence"):
        predict_thresholds(values + .1, .1, model, scorecasts=cache)
    with pytest.raises(ValueError, match="mismatched alpha"):
        predict_thresholds(values, .2, model, scorecasts=cache)


def test_ar_correction_ignores_lqt_bias_and_initial_quantile():
    pytest.importorskip("cvxpy")
    values = np.array([.2, .8, .1, .7, 1.3])
    model = _model(method="sqt_ar", p_order=1)
    normal = predict_thresholds(values, .1, model)
    altered = predict_thresholds(values, .1, {**model, "bias": 100, "initial_quantile": -50})
    np.testing.assert_array_equal(normal["thresholds"], altered["thresholds"])
    np.testing.assert_array_equal(normal["corrections"][:2], [0., 0.])


@pytest.mark.parametrize("settings", [
    {"p_order": -1}, {"p_order": 1.5}, {"p_order": True},
    {"learning_rate": -1}, {"learning_rate": float("inf")},
    {"learning_rate_schedule": "proportional"}, {"bias": 0},
    {"ar_window_size": 0}, {"decay_exponent": 0},
    {"solver": ""}, {"solver_options": {"solver": "SCS"}},
])
def test_invalid_model_settings_fail_before_running(settings):
    with pytest.raises(ValueError):
        validate_model(settings)


@pytest.mark.parametrize("scores", [[], [float("nan")], [-.1], [[1.]], [True]])
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError, match="scores|Scores"):
        predict_thresholds(scores, .1, _model())


def test_nonfinite_thresholds_raise_instead_of_becoming_full_sets():
    with pytest.raises(RuntimeError, match="overflowed"):
        predict_thresholds([1e308, 1e308, 1.], .1,
                           _model(p_order=1, bias=1, learning_rate=1e308))


def test_ar_solver_failure_is_explicit(monkeypatch):
    cp = pytest.importorskip("cvxpy")
    def fail(*args, **kwargs):
        raise cp.error.SolverError("test failure")
    monkeypatch.setattr(cp.Problem, "solve", fail)
    with pytest.raises(RuntimeError, match="solver failed at score index 1"):
        predict_thresholds([1., 2., 3.], .1, _model(method="sqt_ar", p_order=1))


@pytest.mark.parametrize("order", [0, 1, 2])
def test_short_ar_sequences_do_not_expose_current_score(order):
    pytest.importorskip("cvxpy")
    model = _model(method="sqt_ar", p_order=order)
    first = predict_thresholds([0.1], .1, model)["thresholds"]
    second = predict_thresholds([1000.], .1, model)["thresholds"]
    np.testing.assert_array_equal(first, [0.])
    np.testing.assert_array_equal(second, first)
