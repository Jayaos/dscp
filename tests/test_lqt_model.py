"""Numerical and causal fidelity checks for the published LQT algorithm."""

import numpy as np
import pytest

from baselines.lqt.model import predict_thresholds, validate_model


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
def test_predictions_use_only_strictly_past_scores(order):
    model = _model(p_order=order, learning_rate=.15)
    prefix = np.array([.2, .8, .3, .7, .1])
    a = predict_thresholds(np.r_[prefix, 1., 3.], .1, model)["thresholds"]
    b = predict_thresholds(np.r_[prefix, 300., 1000.], .1, model)["thresholds"]
    # Even prediction at the first changed outcome must be identical.
    np.testing.assert_allclose(a[:len(prefix) + 1], b[:len(prefix) + 1], atol=1e-10)
    c = predict_thresholds(prefix, .1, model)["thresholds"]
    np.testing.assert_allclose(a[:len(prefix)], c, atol=1e-10)


@pytest.mark.parametrize("settings", [
    {"p_order": -1}, {"p_order": 1.5}, {"p_order": True},
    {"learning_rate": -1}, {"learning_rate": float("inf")},
    {"learning_rate_schedule": "proportional"}, {"bias": 0},
    {"decay_exponent": 0}, {"method": "sqt_ar"},
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


@pytest.mark.parametrize("order", [0, 1, 2])
def test_short_sequences_do_not_expose_current_score(order):
    model = _model(p_order=order)
    first = predict_thresholds([0.1], .1, model)["thresholds"]
    second = predict_thresholds([1000.], .1, model)["thresholds"]
    np.testing.assert_array_equal(first, [1. if order == 0 else 0.])
    np.testing.assert_array_equal(second, first)
