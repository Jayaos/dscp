"""Check DistMatch's data boundaries, online protocol, and real process workers."""

import copy
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap

import numpy as np
from omegaconf import OmegaConf
import pytest

from baselines.distmatch.config import split_boundaries, validate_config
from baselines.distmatch.data import prepare_sequence
from baselines.distmatch.run_distmatch import evaluate_sequence, evaluate_sequences, run_distmatch
from utils.plotting import _resolve_logged_interval_endpoints


def config():
    return {
        "seed": 37,
        "num_cores": 1,
        "threads_per_worker": 1,
        "data": {"train_ratio": 0.5, "valid_ratio": 0.2, "test_ratio": 0.3, "normalize": True},
        "model": {
            "past_window_len": 3, "match_threshold": 0.5,
            "n_trees": 2, "qrf_n_estimators": 2, "qrf_max_depth": 2,
            "beta_bins": 3, "target_quantiles": [[0.05, 0.95], [0.1, 0.9]],
        },
    }


def artifact(length=40):
    t = np.arange(length, dtype=float)
    predictions = 40 + t * 0.2
    return {
        "heldout_y": predictions + 2 + np.sin(t * 0.8),
        "heldout_predictions": predictions[:, None],
        # Neither covariates nor the point predictor's fitting data is needed.
        "heldout_x": np.full((length, 2), np.nan),
        "train_y": np.full(25, np.nan),
    }


@pytest.mark.parametrize("length,ratios,expected", [
    (100, (0.5, 0.16, 0.34), (50, 66, 34)),
    (101, (0.5, 0.16, 0.34), (50, 67, 34)),
    (100, (0.6, 0.2, 0.2), (60, 80, 20)),
    (100, (0.29, 0.14, 0.57), (29, 43, 57)),
    (101, (0.66, 0, 0.34), (66, 66, 35)),
])
def test_configurable_boundaries_match_target_indices(length, ratios, expected):
    boundaries = split_boundaries(length, *ratios)
    assert tuple(boundaries[key] for key in ("train_end", "test_start", "test_size")) == expected
    assert sum(boundaries[key] for key in ("train_size", "validation_size", "test_size")) == length


@pytest.mark.parametrize("ratios", [(0, .2, .8), (.5, -.1, .6), (.5, .5, 0), (.5, .2, .4),
                                   (True, 0, 0), (.5, np.nan, .5), (.5, np.inf, .5)])
def test_invalid_ratio_configs_are_rejected(ratios):
    cfg = config()
    cfg["data"].update(zip(("train_ratio", "valid_ratio", "test_ratio"), ratios))
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("workers", [0, -1, 1.5, True, "2"])
def test_invalid_worker_config_is_rejected(workers):
    cfg = config()
    cfg["num_cores"] = workers
    with pytest.raises(ValueError, match="num_cores"):
        evaluate_sequences({"a": artifact()}, cfg)


def test_mixed_shapes_slice_before_subtracting():
    item = artifact(101)
    cfg = config()
    cfg["data"].update(train_ratio=.5, valid_ratio=.16, test_ratio=.34)
    prepared = prepare_sequence(item, cfg)
    residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
    np.testing.assert_array_equal(prepared["train_residuals"], residuals[:50])
    np.testing.assert_array_equal(prepared["warmup_residuals"], residuals[50:67])
    np.testing.assert_array_equal(prepared["residuals"], residuals[67:])
    np.testing.assert_array_equal(prepared["target_indices"], np.arange(67, 101))


def test_poisoned_reserved_test_does_not_change_validation_or_scaler():
    item = artifact()
    baseline = evaluate_sequence("a", item, config(), split="validation")
    changed = copy.deepcopy(item)
    changed["heldout_y"][28:] = np.nan
    changed["heldout_predictions"][28:] = np.inf
    actual = evaluate_sequence("a", changed, config(), split="validation")
    assert actual["evaluation_results"] == baseline["evaluation_results"]
    for name in ("input_mean", "input_std", "initial_memory_size", "final_memory_size"):
        assert actual["metadata"][name] == baseline["metadata"][name]
    expected = item["heldout_y"][:20] - item["heldout_predictions"][:20, 0]
    assert actual["metadata"]["input_mean"] == pytest.approx(expected.mean())
    assert actual["metadata"]["input_std"] == pytest.approx(expected.std())


def test_current_target_does_not_change_issued_interval_and_updates_once():
    item = artifact()
    baseline = evaluate_sequence("a", item, config())
    changed = copy.deepcopy(item)
    changed["heldout_y"][28] += 1000
    changed["heldout_y"][29:] -= 2000
    actual = evaluate_sequence("a", changed, config())
    for pair, expected in baseline["evaluation_results"].items():
        result = actual["evaluation_results"][pair]
        assert result["lower_interval"][0] == expected["lower_interval"][0]
        assert result["upper_interval"][0] == expected["upper_interval"][0]
    metadata = baseline["metadata"]
    assert metadata["initial_memory_size"] == 20 - 3 + 8
    assert metadata["final_memory_size"] == 40 - 3
    assert metadata["diagnostics"]["observed_updates"] == 20


def test_no_validation_partition_and_exact_first_test_target():
    cfg = config()
    cfg["data"].update(train_ratio=.7, valid_ratio=0, test_ratio=.3)
    result = evaluate_sequence("a", artifact(), cfg)
    assert result["metadata"]["validation_replay_size"] == 0
    assert result["metadata"]["target_indices"] == list(range(28, 40))
    with pytest.raises(ValueError, match="valid_ratio"):
        evaluate_sequence("a", artifact(), cfg, split="validation")


def test_serial_and_multicore_sequence_results_are_identical():
    data = {"second": artifact(43), "first": artifact(40)}
    cfg = config()
    serial = evaluate_sequences(data, cfg)
    cfg["num_cores"] = 2  # Exercise the YAML/config value, with no API override.
    parallel = evaluate_sequences(data, cfg)
    assert list(parallel) == list(data)
    for key in data:
        assert parallel[key]["evaluation_results"] == serial[key]["evaluation_results"]
        assert parallel[key]["metadata"]["sequence_seed"] == serial[key]["metadata"]["sequence_seed"]
    alone = evaluate_sequences({"first": data["first"]}, cfg, num_cores=1)
    assert alone["first"]["evaluation_results"] == serial["first"]["evaluation_results"]


def test_native_thread_limits_hold_during_real_qrf_fits():
    # A fresh process is essential: already-imported native pools would hide
    # the regression where sklearn/SciPy first load inside threadpool_limits.
    script = textwrap.dedent('''
        import numpy as np
        from threadpoolctl import threadpool_info
        from baselines.distmatch.model import DistMatchResidualIntervalEstimator
        from baselines.distmatch.run_distmatch import evaluate_sequence

        load = DistMatchResidualIntervalEstimator._load_qrf
        observed = []
        def inspect_load():
            cls = load()
            if not getattr(cls, '_distmatch_test_wrapped', False):
                original_fit = cls.fit
                def fit(self, *args, **kwargs):
                    observed.extend(pool['num_threads'] for pool in threadpool_info())
                    return original_fit(self, *args, **kwargs)
                cls.fit = fit
                cls._distmatch_test_wrapped = True
            return cls
        DistMatchResidualIntervalEstimator._load_qrf = staticmethod(inspect_load)
        cfg = {
            'num_cores': 1, 'threads_per_worker': 1,
            'data': {'train_ratio': .5, 'valid_ratio': .2, 'test_ratio': .3},
            'model': {'past_window_len': 3, 'n_trees': 1, 'qrf_n_estimators': 2},
        }
        evaluate_sequence('a', {'heldout_y': np.sin(np.arange(30)),
                               'heldout_predictions': np.zeros(30)}, cfg)
        assert observed and max(observed) == 1, observed
    ''')
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                               cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_reversed_pair_and_additional_coverage_levels_do_not_change_intervals():
    cfg = config()
    original = evaluate_sequence("a", artifact(), cfg)
    cfg["model"]["target_quantiles"] = [[.95, .05]]
    reversed_pair = evaluate_sequence("a", artifact(), cfg)
    expected = original["evaluation_results"][(.05, .95)]
    actual = reversed_pair["evaluation_results"][(.95, .05)]
    assert actual == expected


def test_runner_saves_original_scale_intervals_metrics_and_config(tmp_path):
    cfg = config()
    path = tmp_path / "forecasts.pkl"
    with path.open("wb") as stream:
        pickle.dump({"a": artifact(40), "b": artifact(43)}, stream)
    cfg["data"]["data_path"] = str(path)
    cfg["saving_dir"] = str(tmp_path / "results")
    cfg["plotting"] = {"plotting": True, "plotting_seq_len": 8}
    yaml_path = tmp_path / "config.yaml"
    OmegaConf.save(config=OmegaConf.create(cfg), f=yaml_path)
    log = run_distmatch(yaml_path)
    directory = tmp_path / "results"
    for name in ("log.pkl", "summary_results.pkl", "resolved_config.yaml", "run_metadata.yaml"):
        assert (directory / name).is_file()
    assert len(list((directory / "plots").glob("*.pdf"))) == 4
    saved = OmegaConf.load(directory / "resolved_config.yaml")
    assert (saved.data.train_ratio, saved.data.valid_ratio, saved.data.test_ratio) == (.5, .2, .3)
    assert saved.num_cores == 1
    with (directory / "summary_results.pkl").open("rb") as stream:
        summary = pickle.load(stream)
    for pair in (.05, .95), (.1, .9):
        for entry in log.values():
            result = entry["evaluation_results"][pair]
            lower, upper = _resolve_logged_interval_endpoints(result)
            pred = np.asarray(result["target_predictions"])
            np.testing.assert_allclose(lower, pred + result["lower_residual_quantile"])
            np.testing.assert_allclose(upper, pred + result["upper_residual_quantile"])
            y = np.asarray(result["target_y"])
            alpha = 1 - (pair[1] - pair[0])
            score = upper - lower + (2 / alpha) * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
            np.testing.assert_allclose(result["winkler_score"], score)
        assert summary[pair]["avg_interval_width_mean"] == pytest.approx(np.mean([
            entry["evaluation_results"][pair]["avg_interval_width"] for entry in log.values()
        ]))
