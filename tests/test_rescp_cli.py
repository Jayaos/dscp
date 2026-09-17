import contextlib
import io
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from sbatch.sbatch_run_rescp import run_rescp as cli
from sbatch.sbatch_run_tuning import run_rescp_tuning as tuning


REPO_ROOT = Path(__file__).resolve().parents[1]


def _evaluation(coverage=True, score=2.0, length=3):
    return {
        "evaluation_results": {
            (0.05, 0.95): {
                "coverage": [coverage] * length,
                "interval_width": [1.0] * length,
                "winkler_score": [score] * length,
            }
        },
        "metadata": {"evaluation_split": "validation"},
    }


class ResCPCLITests(unittest.TestCase):
    def test_all_nine_presets_resolve_artifacts_and_seed_paths(self):
        paths = {
            "air": ("air-10_prediction", "air-10"),
            "solar": ("solar_prediction", "nsdb-60m"),
            "sapflux": ("sapflux-solo3-large", "sapflux-solo3-large"),
        }
        outputs = set()
        for dataset, (folder, artifact) in paths.items():
            for predictor in cli.BASE_PREDICTORS:
                with self.subTest(dataset=dataset, predictor=predictor):
                    args = cli.build_parser().parse_args([
                        "--dataset", dataset, "--base-predictor", predictor, "--seed", "71",
                    ])
                    config_path, config = cli.resolve_config(args)
                    self.assertTrue(config_path.is_file())
                    self.assertEqual(Path(config.data.data_path), (
                        REPO_ROOT / "data" / folder / predictor
                        / "{}_{}_data.pkl".format(predictor, artifact)
                    ))
                    self.assertEqual(config.seed, 71)
                    self.assertIn("seed_71", config.saving_dir)
                    self.assertEqual(config.model.prediction_step, 1)
                    self.assertNotIn("validation_ratio", config.data)
                    self.assertAlmostEqual(
                        config.data.calibration_ratio + config.data.test_ratio, 1.0,
                    )
                    outputs.add(config.saving_dir)
        self.assertEqual(len(outputs), 9)

    def test_dry_run_supports_missing_artifact_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_be_created"
            missing = Path(directory) / "missing.pkl"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                config = cli.main([
                    "--dataset", "sapflux", "--base-predictor", "chronos", "--dry-run",
                    "--data-path", str(missing), "--output-dir", str(output),
                ])
            self.assertIn("Artifact available: False", stdout.getvalue())
            self.assertEqual(Path(config.data.data_path), missing)
            self.assertFalse(output.exists())

    def test_preset_and_config_arguments_are_unambiguous(self):
        for arguments in ([], ["--dataset", "air"],
                          ["custom.yaml", "--dataset", "air", "--base-predictor", "lr"]):
            with self.subTest(arguments=arguments):
                args = cli.build_parser().parse_args(arguments)
                with self.assertRaises(ValueError):
                    cli.resolve_config(args)

    def test_seed_validation_matches_estimator_bounds(self):
        parsers = (
            (cli.build_parser(), ["--dataset", "air", "--base-predictor", "lr"]),
            (tuning.build_parser(), ["--base-config", "base.yaml", "--grid-config", "grid.yaml",
                                    "--save-dir", "results"]),
        )
        for parser, required in parsers:
            for seed in ("-1", "4294967296", "1.5"):
                with self.subTest(parser=parser.description, seed=seed), \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(required + ["--seed", seed])
            for seed in (0, 2**32 - 1):
                self.assertEqual(parser.parse_args(required + ["--seed", str(seed)]).seed, seed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            config = OmegaConf.load(REPO_ROOT / "configs/rescp_configs/rescp_lr_air_config.yaml")
            for seed in (True, 2.5, 2**32):
                config.seed = seed
                OmegaConf.save(config, path)
                args = cli.build_parser().parse_args([str(path), "--dry-run"])
                with self.subTest(config_seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                    cli.resolve_config(args)

    def test_direct_and_module_help_work_from_repository_root(self):
        commands = [
            ["sbatch/sbatch_run_rescp/run_rescp.py", "--help"],
            ["-m", "sbatch.sbatch_run_rescp.run_rescp", "--help"],
            ["sbatch/sbatch_run_tuning/run_rescp_tuning.py", "--help"],
            ["-m", "sbatch.sbatch_run_tuning.run_rescp_tuning", "--help"],
        ]
        for arguments in commands:
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable] + arguments, cwd=REPO_ROOT,
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("ResCP", completed.stdout)


class ResCPTuningTests(unittest.TestCase):
    def _configs(self, directory, threshold=-0.01):
        base = OmegaConf.load(REPO_ROOT / "configs/rescp_configs/rescp_lr_air_config.yaml")
        base.data.calibration_ratio = 0.66
        base.data.test_ratio = 0.34
        base.data.pop("validation_ratio", None)
        base.model.reservoir_size = 4
        base.model.temperature = 0.1
        base.model.calibration_size = 8
        base.model.beta_bins = 4
        base.model.connectivity = 1.0
        base.plotting.plotting = False
        base.run_label = "temperature_${model.temperature}"
        base_path = Path(directory) / "base.yaml"
        grid_path = Path(directory) / "grid.yaml"
        OmegaConf.save(base, base_path)
        OmegaConf.save(OmegaConf.create({
            "grid": {"model.temperature": [0.1, 0.2]},
            "tuning": {"num_sequences": 1, "delta_threshold": threshold},
        }), grid_path)
        return base_path, grid_path

    def test_grid_preserves_interpolation_and_protects_comparison_targets(self):
        base = OmegaConf.create({"model": {"temperature": 0.1}, "label": "${model.temperature}"})
        configs = list(tuning.iter_trial_configs(base, {"model.temperature": [0.2, 0.3]}))
        self.assertEqual([float(config.label) for config, _ in configs], [0.2, 0.3])
        self.assertEqual(base.model.temperature, 0.1)
        for key in ("data.calibration_ratio", "data.validation_ratio", "data.test_ratio",
                    "data.data_path", "model.target_quantiles", "model.prediction_step", "seed",
                    "tuning.model_selection_valid_ratio"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Keep the artifact"):
                    list(tuning.iter_trial_configs(base, {key: [1]}))

    def test_tuning_rejects_noninteger_seed_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            for seed in (True, 1.2, 2**32, -1):
                with self.subTest(seed=seed), patch.object(tuning, "load_data") as load:
                    with self.assertRaisesRegex(ValueError, "seed"):
                        tuning.run_tuning(base_path, grid_path, Path(directory) / "output", seed=seed)
                    load.assert_not_called()

    def test_tuning_rejects_invalid_inner_ratio_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            for ratio in (0, 1, -0.1, 1.1, float("nan"), float("inf"), True, "0.2", None):
                grid_config = OmegaConf.load(grid_path)
                grid_config.tuning.model_selection_valid_ratio = ratio
                OmegaConf.save(grid_config, grid_path)
                with self.subTest(ratio=ratio), patch.object(tuning, "load_data") as load:
                    with self.assertRaisesRegex(ValueError, "model_selection_valid_ratio"):
                        tuning.run_tuning(base_path, grid_path, Path(directory) / "output")
                    load.assert_not_called()

    def test_inner_ratio_is_resolved_once_and_grid_tuning_overrides_base_options(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            base_config = OmegaConf.load(base_path)
            base_config.tuning = {
                "model_selection_valid_ratio": "${model.temperature}",
                "num_sequences": 5,
            }
            OmegaConf.save(base_config, base_path)
            ratios = []

            def evaluate(data, config, split, num_cores):
                ratios.append(float(config.tuning.model_selection_valid_ratio))
                self.assertEqual(config.tuning.num_sequences, 1)
                return {"series": _evaluation()}

            with patch.object(tuning, "load_data", return_value={"series": {}}), \
                    patch.object(tuning, "evaluate_sequences", side_effect=evaluate), \
                    contextlib.redirect_stdout(io.StringIO()):
                payload = tuning.run_tuning(base_path, grid_path, Path(directory) / "output")
            self.assertEqual(ratios, [0.1, 0.1])
            self.assertEqual(payload["model_selection_valid_ratio"], 0.1)
            best = OmegaConf.load(payload["best_config_path"])
            self.assertEqual(best.tuning.model_selection_valid_ratio, 0.1)

    def test_validation_only_search_uses_same_seed_and_exports_best_config(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            calls = []

            def evaluate(data, config, split, num_cores):
                calls.append((split, int(config.seed), str(config.run_label), num_cores))
                self.assertEqual(config.tuning.model_selection_valid_ratio, 0.2)
                return {"series": _evaluation(score=3.0 - float(config.model.temperature))}

            with patch.object(tuning, "load_data", return_value={"series": {}}), \
                    patch.object(tuning, "evaluate_sequences", side_effect=evaluate), \
                    contextlib.redirect_stdout(io.StringIO()):
                output = Path(directory) / "tuning"
                payload = tuning.run_tuning(base_path, grid_path, output, seed=17, num_cores=2)

            self.assertEqual(calls, [
                ("validation", 17, "temperature_0.1", 2),
                ("validation", 17, "temperature_0.2", 2),
            ])
            self.assertEqual(payload["selection_status"], "selected")
            self.assertEqual(payload["evaluation_region"], "calibration_tail")
            self.assertFalse(payload["final_test_evaluated"])
            self.assertEqual(payload["model_selection_valid_ratio"], 0.2)
            best = OmegaConf.load(output / "best_config.yaml")
            self.assertEqual(best.model.temperature, 0.2)
            self.assertEqual(best.seed, 17)
            self.assertEqual(best.tuning.model_selection_valid_ratio, 0.2)
            self.assertNotIn("validation_ratio", best.data)
            self.assertEqual(Path(best.saving_dir), output / "final_test")
            self.assertFalse((output / "final_test").exists())
            for relative in ("tuning_results.pkl", "trial_0001/result.pkl",
                             "trial_0001/resolved_config.yaml", "trial_0002/result.pkl"):
                self.assertTrue((output / relative).is_file(), relative)

    def test_outer_split_and_quantile_interpolations_stay_fixed_across_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            base_config = OmegaConf.load(base_path)
            base_config.data.calibration_ratio = "${model.temperature}"
            base_config.data.test_ratio = 0.9
            base_config.model.target_quantiles = [["${model.temperature}", 0.95]]
            OmegaConf.save(base_config, base_path)
            captured = []

            def evaluate(data, config, split, num_cores):
                captured.append((
                    float(config.data.calibration_ratio),
                    float(config.model.target_quantiles[0][0]),
                    float(config.model.temperature),
                ))
                result = _evaluation()
                result["evaluation_results"][(0.1, 0.95)] = result["evaluation_results"].pop((0.05, 0.95))
                return {"series": result}

            with patch.object(tuning, "load_data", return_value={"series": {}}), \
                    patch.object(tuning, "evaluate_sequences", side_effect=evaluate), \
                    contextlib.redirect_stdout(io.StringIO()):
                tuning.run_tuning(base_path, grid_path, Path(directory) / "output")
            self.assertEqual(captured, [(0.1, 0.1, 0.1), (0.1, 0.1, 0.2)])

    def test_no_eligible_trial_reports_every_result_and_clears_stale_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            output = Path(directory) / "tuning"
            output.mkdir()
            (output / "best_config.yaml").write_text("stale: true\n", encoding="utf-8")
            with patch.object(tuning, "load_data", return_value={"series": {}}), \
                    patch.object(tuning, "evaluate_sequences", return_value={"series": _evaluation(False)}), \
                    contextlib.redirect_stdout(io.StringIO()):
                payload = tuning.run_tuning(base_path, grid_path, output)
            self.assertEqual(payload["selection_status"], "no_eligible_trials")
            self.assertEqual(len(payload["all_trials"]), 2)
            self.assertEqual(payload["top_trials"], [])
            self.assertIsNone(payload["best_config_path"])
            self.assertFalse((output / "best_config.yaml").exists())
            self.assertTrue((output / "tuning_results.pkl").is_file())

    def test_null_threshold_disables_coverage_filter_and_sequences_have_equal_weight(self):
        log = {"short": _evaluation(False, score=1.0, length=1),
               "long": _evaluation(True, score=3.0, length=9)}
        result = tuning.aggregate_validation_results(log, [[0.05, 0.95]], delta_threshold=None)
        self.assertTrue(result["coverage_eligible"])
        self.assertEqual(result["selection_score"], 2.0)
        self.assertEqual(result["pair_metrics"]["(0.05, 0.95)"]["avg_coverage"], 0.5)
        self.assertFalse(tuning.aggregate_validation_results(log, [[0.05, 0.95]])["coverage_eligible"])

    def test_real_tuning_rankings_ignore_reserved_test_nan_values(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory, threshold=None)
            time = np.arange(100, dtype=float)
            original = {
                "series": {
                    "heldout_y": np.sin(time * 0.17) + 0.01 * time,
                    "heldout_predictions": np.cos(time * 0.1) * 0.2,
                }
            }
            payloads = []
            for variant in ("original", "test_nan"):
                data = {"series": {key: value.copy() for key, value in original["series"].items()}}
                if variant == "test_nan":
                    # The outer calibration prefix has floor(100 * .66) = 66 rows.
                    for values in data["series"].values():
                        values[66:] = np.nan
                artifact = Path(directory) / (variant + ".pkl")
                with artifact.open("wb") as stream:
                    pickle.dump(data, stream)
                config = OmegaConf.load(base_path)
                config.data.data_path = str(artifact)
                OmegaConf.save(config, base_path)
                with contextlib.redirect_stdout(io.StringIO()):
                    payloads.append(tuning.run_tuning(
                        base_path, grid_path, Path(directory) / variant, seed=17,
                    ))
            selection_metadata = payloads[0]["all_trials"][0]["result"]["sequence_results"]["series"]["metadata"]
            self.assertEqual(selection_metadata["target_indices"], list(range(52, 66)))
            # The exported final-run config retains its tuning provenance but
            # consumes all 66 calibration rows when the ordinary runner is used.
            from baselines.rescp.data import prepare_sequence
            best = OmegaConf.load(payloads[0]["best_config_path"])
            final_data = prepare_sequence(original["series"], best, split="test")
            np.testing.assert_array_equal(final_data["target_indices"], np.arange(66, 100))
            self.assertEqual(len(final_data["calibration_residuals"]), 66)
            self.assertEqual(len(final_data["warmup_residuals"]), 0)
            for left, right in zip(payloads[0]["all_trials"], payloads[1]["all_trials"]):
                for metric in ("pair_metrics", "selection_score", "coverage_eligible"):
                    self.assertEqual(left["result"][metric], right["result"][metric])
                left_sequence = left["result"]["sequence_results"]["series"]
                right_sequence = right["result"]["sequence_results"]["series"]
                self.assertEqual(left_sequence["pair_metrics"], right_sequence["pair_metrics"])
                # Runtime is observational metadata, not a deterministic model output.
                for key, value in left_sequence["metadata"].items():
                    if not key.endswith("_seconds"):
                        self.assertEqual(value, right_sequence["metadata"][key])
            self.assertEqual(
                [trial["grid_values"] for trial in payloads[0]["top_trials"]],
                [trial["grid_values"] for trial in payloads[1]["top_trials"]],
            )


if __name__ == "__main__":
    unittest.main()
