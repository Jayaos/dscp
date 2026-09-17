"""SPCI tuning must select on a chronological part of nominal training only."""

import contextlib
import io
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from sbatch.sbatch_run_tuning import run_spci_tuning as tuning


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR = (0.75, 0.25)


class EmpiricalForest:
    """A small deterministic forest substitute; real splitting and metrics run."""

    def __init__(self, quantiles):
        self.quantiles = np.asarray(quantiles)
        self.fit_features = None
        self.fit_targets = None
        self.prediction_features = None

    def fit(self, features, targets):
        self.fit_features = np.asarray(features).copy()
        self.fit_targets = np.asarray(targets).copy()
        if not np.isfinite(self.fit_features).all() or not np.isfinite(self.fit_targets).all():
            raise AssertionError("Non-finite reserved values reached fitting.")
        return self

    def predict(self, features):
        self.prediction_features = np.asarray(features).copy()
        if not np.isfinite(self.prediction_features).all():
            raise AssertionError("Non-finite reserved values reached prediction.")
        values = np.quantile(self.fit_targets, self.quantiles)
        return np.repeat(values[:, None], len(features), axis=1).astype(np.float32)


def _artifact():
    index = np.arange(100, dtype=np.float32)
    return {
        key: {
            "heldout_x": np.stack((index, np.sin(index / 11)), axis=1),
            "heldout_y": 0.1 * index + np.sin(index / 3) + offset,
            "heldout_predictions": 0.1 * index,
        }
        for key, offset in (("zeta", 0.3), ("alpha", -0.5), ("mu", 0.7))
    }


def _summary(score=2.0, coverage=True, length=10):
    pair_metrics, selection_score, eligible = tuning.summarize_evaluation_results(
        {PAIR: {
            "coverage": [coverage] * length,
            "interval_width": [1.0] * length,
            "winkler_score": [score] * length,
        }},
        [PAIR],
        delta_threshold=0.0,
    )
    return {
        "pair_metrics": pair_metrics,
        "selection_score": selection_score,
        "positive_delta_coverage": eligible,
        "num_tuning_evaluation_samples": length,
        "final_test_evaluated": False,
    }


class SPCITuningTests(unittest.TestCase):
    def _configs(self, directory, *, grid=None, num_sequences=1, ratio=0.2, threshold=-1.0):
        directory = Path(directory)
        artifact = directory / "lr_toy_data.pkl"
        with artifact.open("wb") as stream:
            pickle.dump(_artifact(), stream)
        base = OmegaConf.create({
            "seed": 17,
            "num_cores": 1,
            "data": {
                "data_path": str(artifact),
                "train_ratio": 0.5,
                "valid_ratio": 0.16,
                "normalize": True,
            },
            "model": {
                "target_quantiles": [list(PAIR)],
                "prediction_step": 1,
                "window_size": 3,
                "n_estimators": 2,
                "max_depth": 2,
                "criterion": "squared_error",
            },
            "plotting": {"plotting": False},
            "saving_dir": str(directory / "ordinary_run"),
        })
        base_path, grid_path = directory / "base.yaml", directory / "grid.yaml"
        OmegaConf.save(base, base_path)
        OmegaConf.save(OmegaConf.create({
            "grid": grid or {"model.window_size": [3, 5]},
            "tuning": {
                "num_sequences": num_sequences,
                "delta_threshold": threshold,
                "model_selection_valid_ratio": ratio,
            },
        }), grid_path)
        return base_path, grid_path

    @contextlib.contextmanager
    def _forests(self):
        forests = []

        def factory(config, n_train_samples, quantiles):
            forest = EmpiricalForest(quantiles)
            forest.expected_fit_count = n_train_samples
            forests.append(forest)
            return forest

        with patch.object(tuning, "build_quantile_forest", side_effect=factory):
            yield forests

    def test_config_predictor_routes_artifact_and_distinct_output_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "base.yaml"
            for predictor in ("lr", "lstm", "chronos"):
                with self.subTest(predictor=predictor):
                    OmegaConf.save(OmegaConf.create({
                        "base_predictor": f" {predictor.upper()} ",
                        "data": {"data_path": "data/${base_predictor}/${base_predictor}_toy_data.pkl"},
                        "model": {"max_depth": 7},
                        "saving_dir": "results/spci_${base_predictor}_toy",
                    }), source)
                    config, selected_path, output = tuning.resolve_tuning_inputs(
                        source, {}, directory / "spci_{base_predictor}_toy"
                    )
                    self.assertEqual(selected_path, source.resolve())
                    self.assertEqual(config.base_predictor, predictor)
                    self.assertEqual(config.model.max_depth, 7)
                    self.assertEqual(Path(config.data.data_path),
                                     REPO_ROOT / "data" / predictor / f"{predictor}_toy_data.pkl")
                    self.assertEqual(Path(config.saving_dir), Path("results") / f"spci_{predictor}_toy")
                    self.assertEqual(output, directory / f"spci_{predictor}_toy")

    def test_existing_configs_resolve_predictor_placeholder_without_changing_dataset(self):
        config_dir = REPO_ROOT / "configs" / "spci_configs"
        configs = sorted(
            source
            for predictor in ("lr", "lstm", "chronos")
            for source in config_dir.glob(f"spci_{predictor}_*_config.yaml")
        )
        self.assertTrue(configs)
        for source in configs:
            with self.subTest(config=source.name):
                original = OmegaConf.load(source)
                predictor = Path(str(original.data.data_path)).stem.split("_", 1)[0]
                config, selected_path, output = tuning.resolve_tuning_inputs(
                    source, {}, REPO_ROOT / "unused" / "spci_{base_predictor}"
                )
                self.assertEqual(selected_path, source.resolve())
                self.assertEqual(Path(config.data.data_path),
                                 (REPO_ROOT / original.data.data_path).resolve())
                self.assertEqual(output, REPO_ROOT / "unused" / f"spci_{predictor}")

    def test_grid_rejects_data_split_coverage_and_unknown_changes_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            for dotted_key in (
                "data.train_ratio", "data.valid_ratio", "data.data_path", "data.normalize",
                "model.target_quantiles", "model.prediction_step", "seed", "num_cores",
                "tuning.model_selection_valid_ratio", "model.unknown_setting",
            ):
                with self.subTest(key=dotted_key):
                    base_path, grid_path = self._configs(directory, grid={dotted_key: [1]})
                    with patch.object(tuning, "load_data") as load, self.assertRaises(ValueError):
                        tuning.run_tuning(base_path, grid_path, Path(directory) / "tuning")
                    load.assert_not_called()

    def test_invalid_tuning_fraction_and_seed_fail_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            for ratio in (0.0, 1.0, -0.2, float("nan"), float("inf"), True):
                with self.subTest(ratio=ratio):
                    base_path, grid_path = self._configs(directory, ratio=ratio)
                    with patch.object(tuning, "load_data") as load, \
                            self.assertRaisesRegex(ValueError, "model_selection_valid_ratio"):
                        tuning.run_tuning(base_path, grid_path, Path(directory) / "tuning")
                    load.assert_not_called()
            base_path, grid_path = self._configs(directory)
            for kwargs in ({"seed": -1}, {"seed": 2**32 - 1}, {"seed": True},
                           {"top_k": 0}, {"top_k": 1.5}):
                with self.subTest(kwargs=kwargs):
                    with patch.object(tuning, "load_data") as load, self.assertRaises(ValueError):
                        tuning.run_tuning(base_path, grid_path, Path(directory) / "tuning", **kwargs)
                    load.assert_not_called()

    def test_controllable_ratio_is_fraction_of_nominal_train_and_scores_only_that_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            for ratio, train_end, evaluation_size in ((0.2, 40, 10), (0.4, 30, 20)):
                with self.subTest(ratio=ratio):
                    base_path, grid_path = self._configs(directory, ratio=ratio)
                    with self._forests() as forests, contextlib.redirect_stdout(io.StringIO()):
                        payload = tuning.run_tuning(
                            base_path, grid_path, Path(directory) / f"tuning_{ratio}"
                        )
                    self.assertEqual(payload["tuning_protocol"]["model_selection_valid_ratio"], ratio)
                    self.assertEqual(len(forests), 2)
                    for window, forest, record in zip((3, 5), forests, payload["all_trials"]):
                        self.assertEqual(len(forest.fit_targets), train_end - window)
                        self.assertEqual(forest.expected_fit_count, train_end - window)
                        self.assertEqual(len(forest.prediction_features), evaluation_size)
                        result = record["result"]["sequence_results"]["alpha"]
                        self.assertEqual(result["num_tuning_evaluation_samples"], evaluation_size)
                        self.assertFalse(result["final_test_evaluated"])
                        self.assertTrue(np.isfinite(result["selection_score"]))
                    self.assertFalse(payload["final_test_evaluated"])

    def test_reserved_nominal_validation_and_test_cannot_change_scores_or_forest_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory, num_sequences="all")
            base = OmegaConf.load(base_path)
            observations = []
            for poison_reserved in (False, True):
                artifact = _artifact()
                if poison_reserved:
                    for item in artifact.values():
                        item["heldout_x"][50:] = np.nan
                        item["heldout_y"][50:] = np.nan
                        item["heldout_predictions"][50:] = np.inf
                with Path(base.data.data_path).open("wb") as stream:
                    pickle.dump(artifact, stream)
                with self._forests() as forests, contextlib.redirect_stdout(io.StringIO()):
                    payload = tuning.run_tuning(
                        base_path, grid_path, Path(directory) / f"tuning_{poison_reserved}"
                    )
                observations.append((payload, forests))
            before, after = observations
            self.assertEqual([trial["result"] for trial in before[0]["all_trials"]],
                             [trial["result"] for trial in after[0]["all_trials"]])
            self.assertEqual([trial["trial_index"] for trial in before[0]["top_trials"]],
                             [trial["trial_index"] for trial in after[0]["top_trials"]])
            for first, second in zip(before[1], after[1]):
                np.testing.assert_array_equal(first.fit_features, second.fit_features)
                np.testing.assert_array_equal(first.fit_targets, second.fit_targets)
                np.testing.assert_array_equal(first.prediction_features, second.prediction_features)

    def test_selected_training_suffix_changes_scores_without_changing_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory, grid={"model.window_size": [3]})
            base = OmegaConf.load(base_path)
            observations = []
            for shift in (0.0, 100.0):
                artifact = _artifact()
                artifact["alpha"]["heldout_y"][40:50] += shift
                with Path(base.data.data_path).open("wb") as stream:
                    pickle.dump(artifact, stream)
                with self._forests() as forests, contextlib.redirect_stdout(io.StringIO()):
                    payload = tuning.run_tuning(
                        base_path, grid_path, Path(directory) / f"tuning_{shift}"
                    )
                observations.append((payload["all_trials"][0]["result"]["selection_score"], forests[0]))
            self.assertGreater(observations[1][0], observations[0][0])
            np.testing.assert_array_equal(observations[0][1].fit_targets, observations[1][1].fit_targets)

    def test_artifacts_export_effective_seed_and_unique_reusable_final_run_config(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            output = Path(directory) / "tuning"
            with self._forests(), contextlib.redirect_stdout(io.StringIO()):
                payload = tuning.run_tuning(base_path, grid_path, output, seed=80, top_k=1)
            self.assertEqual(payload["method"], "spci")
            self.assertEqual(len(payload["top_trials"]), 1)
            self.assertEqual(payload["num_trials"], 2)
            with (output / "tuning_results.pkl").open("rb") as stream:
                self.assertEqual(pickle.load(stream), payload)
            for index, record in enumerate(payload["all_trials"], 1):
                trial_dir = output / f"trial_{index:04d}"
                saved = OmegaConf.load(trial_dir / "resolved_config.yaml")
                self.assertEqual(saved.seed, 80 + index)
                self.assertEqual(record["seed"], 80 + index)
                self.assertEqual(OmegaConf.to_container(saved, resolve=True), record["resolved_config"])
                self.assertEqual(Path(saved.saving_dir), trial_dir / "final_run")
                self.assertFalse(Path(saved.saving_dir).exists())
                self.assertEqual(saved.data.data_path, OmegaConf.load(base_path).data.data_path)
                self.assertEqual((saved.data.train_ratio, saved.data.valid_ratio), (0.5, 0.16))
                self.assertEqual(saved.model.window_size, (3, 5)[index - 1])
                with (trial_dir / "result.pkl").open("rb") as stream:
                    self.assertEqual(pickle.load(stream), record)

    def test_coverage_filter_precedes_score_with_stable_ties_and_empty_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(
                directory, grid={"model.n_estimators": [1, 2, 3, 4]}, threshold=0.0
            )

            def result(config, sequence_item, normalization_params):
                trial = config.model.n_estimators
                return _summary(score={1: 2.0, 2: 1.0, 3: 1.0, 4: 0.0}[trial], coverage=trial != 4)

            with patch.object(tuning, "_run_single_trial", side_effect=result), \
                    contextlib.redirect_stdout(io.StringIO()):
                payload = tuning.run_tuning(base_path, grid_path, Path(directory) / "tuning")
            self.assertEqual([trial["trial_index"] for trial in payload["top_trials"]], [2, 3, 1])
            self.assertEqual(payload["num_positive_delta_coverage_trials"], 3)
            with patch.object(tuning, "_run_single_trial", return_value=_summary(coverage=False)), \
                    contextlib.redirect_stdout(io.StringIO()):
                rejected = tuning.run_tuning(base_path, grid_path, Path(directory) / "tuning")
            self.assertEqual(rejected["top_trials"], [])
            self.assertEqual(rejected["num_positive_delta_coverage_trials"], 0)

    def test_aggregation_weights_sequences_equally_and_requires_every_sequence(self):
        short = _summary(score=2.0, coverage=False, length=2)
        long = _summary(score=8.0, coverage=True, length=100)
        result = tuning.aggregate_sequence_results({"short": short, "long": long}, [PAIR])
        self.assertEqual(result["selection_score"], 5.0)
        self.assertEqual(result["pair_metrics"][str(PAIR)]["avg_coverage"], 0.5)
        self.assertFalse(result["positive_delta_coverage"])
        self.assertNotIn("mean_best_epoch", result)
        self.assertNotIn("mean_best_valid_loss", result)

    def test_coverage_threshold_is_strict_and_applies_to_every_quantile_pair(self):
        second_pair = (0.9, 0.1)
        values = {
            PAIR: {"coverage": [True, False], "interval_width": [1, 1], "winkler_score": [2, 2]},
            second_pair: {"coverage": [True, True], "interval_width": [2, 2], "winkler_score": [4, 4]},
        }
        _, score, eligible = tuning.summarize_evaluation_results(values, [PAIR, second_pair], 0.0)
        self.assertEqual(score, 3.0)
        self.assertFalse(eligible)  # The first pair has coverage gap exactly zero.
        self.assertTrue(tuning.summarize_evaluation_results(values, [PAIR, second_pair], -0.01)[2])

    def test_sequence_key_overrides_all_and_numeric_selection_uses_sorted_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            for count, kwargs, expected in (
                ("all", {"sequence_index": 99}, ["alpha", "mu", "zeta"]),
                ("all", {"sequence_key": "zeta", "sequence_index": 99}, ["zeta"]),
                (2, {"sequence_index": 1}, ["mu", "zeta"]),
            ):
                with self.subTest(count=count, kwargs=kwargs):
                    base_path, grid_path = self._configs(
                        directory, grid={"model.window_size": [3]}, num_sequences=count
                    )
                    with self._forests() as forests, contextlib.redirect_stdout(io.StringIO()):
                        payload = tuning.run_tuning(
                            base_path, grid_path, Path(directory) / "tuning", **kwargs
                        )
                    self.assertEqual(payload["sequence_keys"], expected)
                    self.assertEqual(len(forests), len(expected))
                    self.assertEqual(list(payload["all_trials"][0]["result"]["sequence_results"]), expected)

    def test_cli_forwards_the_shared_tuning_arguments(self):
        args = SimpleNamespace(
            base_config=Path("base.yaml"), grid_config=Path("grid.yaml"),
            save_dir=Path("results"), sequence_key="station", sequence_index=4,
            top_k=2, seed=50,
        )
        with patch.object(tuning, "parse_args", return_value=args) as parse, \
                patch.object(tuning, "run_tuning") as run:
            tuning.main()
        parse.assert_called_once_with("spci")
        run.assert_called_once_with(
            args.base_config, args.grid_config, args.save_dir,
            sequence_key="station", sequence_index=4, top_k=2, seed=50,
        )


if __name__ == "__main__":
    unittest.main()
