import contextlib
import copy
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from sbatch.sbatch_run_tuning import run_distmatch_tuning as tuning


PAIR = (0.1, 0.9)


def _evaluation(coverage=True, score=2.0, length=3):
    return {
        "evaluation_results": {
            PAIR: {
                "coverage": [coverage] * length,
                "interval_width": [1.0] * length,
                "winkler_score": [score] * length,
            }
        },
        "metadata": {"split": "validation"},
    }


class DistMatchTuningTests(unittest.TestCase):
    def _configs(self, directory, threshold=-0.01, num_cores=3):
        directory = Path(directory)
        base = OmegaConf.create({
            "seed": 17,
            "num_cores": num_cores,
            "threads_per_worker": 2,
            "data": {
                "data_path": str(directory / "forecast.pkl"),
                "train_ratio": 0.5,
                "valid_ratio": 0.2,
                "test_ratio": 0.3,
                "normalize": True,
            },
            "model": {
                "target_quantiles": [list(PAIR)],
                "prediction_step": 1,
                "past_window_len": 3,
                "match_threshold": 0.1,
                "n_trees": 1,
                "bagging_ratio": 1.0,
                "beta_bins": 3,
                "qrf_n_estimators": 2,
                "qrf_max_depth": 2,
                "min_samples_per_node": 4,
                "use_beta_search": True,
            },
            "plotting": {"plotting": False},
            "saving_dir": str(directory / "ordinary_run"),
            "run_label": "threshold_${model.match_threshold}",
        })
        base_path, grid_path = directory / "base.yaml", directory / "grid.yaml"
        OmegaConf.save(base, base_path)
        OmegaConf.save(OmegaConf.create({
            "grid": {"model.match_threshold": [0.1, 0.2]},
            "tuning": {"num_sequences": 1, "delta_threshold": threshold},
        }), grid_path)
        return base_path, grid_path

    def test_grid_keeps_comparison_settings_fixed_and_resolves_updated_labels(self):
        base = OmegaConf.create({
            "model": {"match_threshold": 0.1},
            "label": "${model.match_threshold}",
        })
        candidates = list(tuning.iter_trial_configs(base, {"model.match_threshold": [0.2, 0.3]}))
        self.assertEqual([float(config.label) for config, _ in candidates], [0.2, 0.3])
        self.assertEqual(base.model.match_threshold, 0.1)
        for key in (
            "data.train_ratio", "data.valid_ratio", "data.test_ratio", "data.data_path",
            "data.normalize", "model.target_quantiles", "model.prediction_step", "seed",
            "num_cores", "threads_per_worker", "model.unknown_setting",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Keep the artifact"):
                list(tuning.iter_trial_configs(base, {key: [1]}))
        base.data = {"valid_ratio": "${model.match_threshold}"}
        with self.assertRaisesRegex(ValueError, "interpolation"):
            list(tuning.iter_trial_configs(base, {"model.match_threshold": [0.2]}))

    def test_validation_only_dispatch_preserves_config_workers_splits_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            output = Path(directory) / "tuning"
            captured = []

            def evaluate(data, config, split, num_cores):
                self.assertEqual(split, "validation")
                self.assertEqual(num_cores, 3)
                self.assertEqual(config.num_cores, 3)
                self.assertEqual(config.threads_per_worker, 2)
                self.assertEqual(config.seed, 17)
                self.assertEqual((config.data.train_ratio, config.data.valid_ratio, config.data.test_ratio),
                                 (0.5, 0.2, 0.3))
                self.assertEqual(list(data), ["station"])
                captured.append(OmegaConf.to_container(config, resolve=True))
                score = 3.0 if config.model.match_threshold == 0.1 else 2.0
                return {"station": _evaluation(score=score)}

            with patch.object(tuning, "load_data", return_value={"station": object()}), \
                    patch.object(tuning, "evaluate_sequences", side_effect=evaluate), \
                    contextlib.redirect_stdout(io.StringIO()):
                payload = tuning.run_tuning(base_path, grid_path, output, top_k=1)
            self.assertEqual(len(captured), 2)
            self.assertEqual(payload["evaluation_split"], "validation")
            self.assertFalse(payload["final_test_evaluated"])
            self.assertEqual(payload["num_cores"], 3)
            self.assertEqual(len(payload["top_trials"]), 1)
            self.assertEqual(payload["top_trials"][0]["trial_index"], 2)
            best = OmegaConf.load(payload["best_config_path"])
            self.assertEqual(best.model.match_threshold, 0.2)
            self.assertEqual(best.run_label, "threshold_0.2")
            self.assertEqual(best.num_cores, 3)
            self.assertEqual(best.threads_per_worker, 2)
            self.assertEqual(best.seed, 17)
            self.assertEqual(OmegaConf.to_container(best.data), captured[0]["data"])
            self.assertEqual(Path(best.saving_dir), output / "final_test")
            self.assertFalse((output / "final_test").exists())
            self.assertTrue((output / "trial_0001" / "resolved_config.yaml").is_file())
            self.assertTrue((output / "trial_0002" / "result.pkl").is_file())
            self.assertTrue((output / "tuning_results.pkl").is_file())

    def test_explicit_worker_and_seed_overrides_are_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            with patch.object(tuning, "load_data", return_value={"station": {}}), \
                    patch.object(tuning, "evaluate_sequences", return_value={"station": _evaluation()}) as evaluate, \
                    contextlib.redirect_stdout(io.StringIO()):
                payload = tuning.run_tuning(base_path, grid_path, Path(directory) / "tuning", num_cores=2, seed=81)
            for call in evaluate.call_args_list:
                self.assertEqual(call.kwargs["num_cores"], 2)
                self.assertEqual(call.args[1].num_cores, 2)
                self.assertEqual(call.args[1].seed, 81)
            best = OmegaConf.load(payload["best_config_path"])
            self.assertEqual(best.num_cores, 2)
            self.assertEqual(best.seed, 81)

    def test_coverage_filter_precedes_ranking_and_no_eligible_run_clears_stale_best(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            output = Path(directory) / "tuning"
            with patch.object(tuning, "load_data", return_value={"station": {}}), \
                    patch.object(tuning, "evaluate_sequences", side_effect=[
                        {"station": _evaluation(coverage=False, score=1.0)},
                        {"station": _evaluation(coverage=True, score=9.0)},
                    ]), contextlib.redirect_stdout(io.StringIO()):
                payload = tuning.run_tuning(base_path, grid_path, output)
            self.assertEqual(payload["num_eligible_trials"], 1)
            self.assertEqual(payload["top_trials"][0]["trial_index"], 2)
            self.assertTrue((output / "best_config.yaml").is_file())
            with patch.object(tuning, "load_data", return_value={"station": {}}), \
                    patch.object(tuning, "evaluate_sequences", return_value={"station": _evaluation(coverage=False)}), \
                    contextlib.redirect_stdout(io.StringIO()):
                payload = tuning.run_tuning(base_path, grid_path, output)
            self.assertEqual(payload["selection_status"], "no_eligible_trials")
            self.assertIsNone(payload["best_config_path"])
            self.assertFalse((output / "best_config.yaml").exists())

    def test_aggregation_weights_sequences_equally_and_can_disable_coverage_filter(self):
        log = {
            "short": _evaluation(coverage=False, score=2.0, length=2),
            "long": _evaluation(coverage=True, score=8.0, length=100),
        }
        result = tuning.aggregate_validation_results(log, [PAIR], delta_threshold=None)
        self.assertEqual(result["selection_score"], 5.0)
        self.assertEqual(result["pair_metrics"][str(PAIR)]["avg_coverage"], 0.5)
        self.assertTrue(result["coverage_eligible"])
        self.assertFalse(tuning.aggregate_validation_results(log, [PAIR])["coverage_eligible"])
        for bad in ([], [np.nan], [np.inf]):
            damaged = copy.deepcopy(log)
            damaged["short"]["evaluation_results"][PAIR]["winkler_score"] = bad
            with self.subTest(values=bad), self.assertRaisesRegex(ValueError, "validation winkler_score"):
                tuning.aggregate_validation_results(damaged, [PAIR])

    def test_cli_defaults_to_config_workers_and_rejects_invalid_counts(self):
        required = ["--base-config", "base.yaml", "--grid-config", "grid.yaml", "--save-dir", "results"]
        parser = tuning.build_parser()
        self.assertIsNone(parser.parse_args(required).num_cores)
        self.assertEqual(parser.parse_args(required + ["--num-cores", "4"]).num_cores, 4)
        for argument, value in (("--num-cores", "0"), ("--num-cores", "1.5"),
                                ("--top-k", "-1"), ("--seed", "4294967296")):
            with self.subTest(argument=argument, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(required + [argument, value])
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            for kwargs in ({"num_cores": True}, {"num_cores": 2.5}, {"seed": -1}, {"seed": 2**32}):
                with self.subTest(kwargs=kwargs), patch.object(tuning, "load_data") as load:
                    with self.assertRaises(ValueError):
                        tuning.run_tuning(base_path, grid_path, Path(directory) / "tuning", **kwargs)
                    load.assert_not_called()

    def test_real_tuning_rankings_ignore_reserved_test_values(self):
        # Exercise the real data/model/runner path with finite and poisoned test
        # suffixes. Both grid searches must consume only the first 70 points.
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory, threshold=None, num_cores=1)
            base = OmegaConf.load(base_path)
            base.threads_per_worker = 1
            OmegaConf.save(base, base_path)
            index = np.arange(100, dtype=np.float64)
            predictions = 10.0 + 0.2 * index
            y = predictions + np.sin(index * 0.7) + 0.05 * (index % 5)
            artifact = {"station": {"heldout_y": y, "heldout_predictions": predictions[:, None]}}
            scores, rankings, pair_metrics = [], [], []
            for run_index in range(2):
                if run_index:
                    artifact["station"]["heldout_y"][70:] = np.nan
                    artifact["station"]["heldout_predictions"][70:] = np.inf
                with Path(base.data.data_path).open("wb") as stream:
                    pickle.dump(artifact, stream)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = tuning.run_tuning(base_path, grid_path, Path(directory) / f"tuning_{run_index}")
                scores.append([trial["result"]["selection_score"] for trial in result["all_trials"]])
                rankings.append([trial["trial_index"] for trial in result["top_trials"]])
                pair_metrics.append([trial["result"]["pair_metrics"] for trial in result["all_trials"]])
                self.assertFalse(result["final_test_evaluated"])
            np.testing.assert_array_equal(scores[0], scores[1])
            self.assertEqual(rankings[0], rankings[1])
            self.assertEqual(pair_metrics[0], pair_metrics[1])


if __name__ == "__main__":
    unittest.main()
