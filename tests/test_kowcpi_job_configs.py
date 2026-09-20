import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_kowcpi import run_kowcpi_job
from sbatch.sbatch_run_tuning import run_kowcpi_tuning_job


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTORS = tuple(run_kowcpi_job.TASKS)
DATASETS = tuple(run_kowcpi_job.DATASET_ARTIFACTS)


def _normal_dry_run_config(output):
    lines = output.splitlines()
    marker_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("Configuration template: ")
    )
    return OmegaConf.create("\n".join(lines[marker_index + 1 :]))


def _tuning_dry_run_config(output):
    lines = output.splitlines()
    yaml_start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("Tuning output: ")
    ) + 1
    yaml_end = next(
        index for index, line in enumerate(lines) if line.startswith("Command: ")
    )
    return OmegaConf.create("\n".join(lines[yaml_start:yaml_end]))


class KOWCPIJobConfigTests(unittest.TestCase):
    def test_all_nine_configs_match_their_source_family_except_paths(self):
        config_dir = REPO_ROOT / "configs" / "kowcpi_configs"

        for dataset in DATASETS:
            source_name = (
                "kowcpi_lstm_sapflux_config.yaml"
                if dataset == "sapflux"
                else "kowcpi_chronos_air_config.yaml"
            )
            source = OmegaConf.to_container(
                OmegaConf.load(config_dir / source_name),
                resolve=True,
            )
            for predictor in PREDICTORS:
                with self.subTest(dataset=dataset, predictor=predictor):
                    config_path = (
                        config_dir
                        / f"kowcpi_{predictor}_{dataset}_config.yaml"
                    )
                    self.assertTrue(config_path.is_file(), config_path)
                    config = OmegaConf.to_container(
                        OmegaConf.load(config_path),
                        resolve=True,
                    )

                    artifact_dir, artifact_name = (
                        run_kowcpi_job.DATASET_ARTIFACTS[dataset]
                    )
                    self.assertEqual(
                        config["data"]["data_path"],
                        (
                            f"./data/{artifact_dir}/{predictor}/"
                            f"{predictor}_{artifact_name}_data.pkl"
                        ),
                    )
                    dataset_label = (
                        "sapflux_solo3_large"
                        if dataset == "sapflux"
                        else dataset
                    )
                    self.assertEqual(
                        config["saving_dir"],
                        f"./results/kowcpi_{predictor}_{dataset_label}/",
                    )

                    source_without_paths = OmegaConf.create(source)
                    config_without_paths = OmegaConf.create(config)
                    del source_without_paths.data.data_path
                    del source_without_paths.saving_dir
                    del config_without_paths.data.data_path
                    del config_without_paths.saving_dir
                    self.assertEqual(
                        OmegaConf.to_container(
                            config_without_paths,
                            resolve=True,
                        ),
                        OmegaConf.to_container(
                            source_without_paths,
                            resolve=True,
                        ),
                    )

    def test_normal_and_tuning_dry_runs_select_all_nine_route_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_root = Path(temp_dir)
            config_dir = fixture_root / "configs" / "kowcpi_configs"
            config_dir.mkdir(parents=True)

            for dataset in DATASETS:
                OmegaConf.save(
                    config=OmegaConf.create(
                        {
                            "grid": {"model.past_window": [3, 5]},
                            "tuning": {
                                "model_selection_valid_ratio": 0.2,
                            },
                        }
                    ),
                    f=config_dir / f"kowcpi_{dataset}_tuning_config.yaml",
                )
                for predictor in PREDICTORS:
                    OmegaConf.save(
                        config=OmegaConf.create(
                            {
                                "route_sentinel": f"{dataset}-{predictor}",
                                "data": {"data_path": "template-data.pkl"},
                                "model": {"prediction_step": 99},
                                "saving_dir": "template-output",
                            }
                        ),
                        f=(
                            config_dir
                            / f"kowcpi_{predictor}_{dataset}_config.yaml"
                        ),
                    )

            normal_output_root = fixture_root / "normal-output"
            tuning_output_root = fixture_root / "tuning-output"
            with (
                patch.object(run_kowcpi_job, "REPO_ROOT", fixture_root),
                patch.object(run_kowcpi_tuning_job, "REPO_ROOT", fixture_root),
            ):
                for dataset in DATASETS:
                    artifact_dir, artifact_name = (
                        run_kowcpi_job.DATASET_ARTIFACTS[dataset]
                    )
                    for task_id, predictor in enumerate(PREDICTORS):
                        with self.subTest(
                            launcher="normal",
                            dataset=dataset,
                            predictor=predictor,
                        ):
                            output = io.StringIO()
                            with contextlib.redirect_stdout(output):
                                run_kowcpi_job.main(
                                    [
                                        dataset,
                                        "--task-id",
                                        str(task_id),
                                        "--output-root",
                                        str(normal_output_root),
                                        "--seed",
                                        "17",
                                        "--dry-run",
                                    ]
                                )
                            output_text = output.getvalue()
                            template_path = (
                                config_dir
                                / f"kowcpi_{predictor}_{dataset}_config.yaml"
                            )
                            self.assertIn(
                                f"Configuration template: {template_path}",
                                output_text,
                            )
                            config = _normal_dry_run_config(output_text)
                            self.assertEqual(
                                config.route_sentinel,
                                f"{dataset}-{predictor}",
                            )
                            self.assertEqual(
                                Path(config.data.data_path),
                                (
                                    fixture_root
                                    / "data"
                                    / artifact_dir
                                    / predictor
                                    / f"{predictor}_{artifact_name}_data.pkl"
                                ),
                            )
                            self.assertEqual(
                                Path(config.saving_dir),
                                normal_output_root.resolve()
                                / dataset
                                / predictor,
                            )
                            self.assertEqual(config.model.prediction_step, 1)
                            self.assertEqual(config.seed, 17)

                        with self.subTest(
                            launcher="tuning",
                            dataset=dataset,
                            predictor=predictor,
                        ):
                            output = io.StringIO()
                            with contextlib.redirect_stdout(output):
                                run_kowcpi_tuning_job.main(
                                    [
                                        dataset,
                                        "--task-id",
                                        str(task_id),
                                        "--output-root",
                                        str(tuning_output_root),
                                        "--seed",
                                        "19",
                                        "--dry-run",
                                    ]
                                )
                            output_text = output.getvalue()
                            template_path = (
                                config_dir
                                / f"kowcpi_{predictor}_{dataset}_config.yaml"
                            )
                            grid_path = (
                                config_dir
                                / f"kowcpi_{dataset}_tuning_config.yaml"
                            )
                            self.assertIn(
                                f"Base template: {template_path.resolve()}",
                                output_text,
                            )
                            self.assertIn(
                                f"Tuning grid: {grid_path.resolve()}",
                                output_text,
                            )
                            self.assertIn(
                                (
                                    "Tuning output: "
                                    f"{tuning_output_root.resolve() / dataset / predictor}"
                                ),
                                output_text,
                            )
                            config = _tuning_dry_run_config(output_text)
                            self.assertEqual(
                                config.route_sentinel,
                                f"{dataset}-{predictor}",
                            )
                            self.assertEqual(
                                Path(config.data.data_path),
                                (
                                    fixture_root
                                    / "data"
                                    / artifact_dir
                                    / predictor
                                    / f"{predictor}_{artifact_name}_data.pkl"
                                ),
                            )
                            self.assertEqual(
                                Path(config.saving_dir),
                                fixture_root
                                / "results"
                                / "kowcpi"
                                / dataset
                                / predictor,
                            )
                            self.assertEqual(config.model.prediction_step, 1)
                            self.assertEqual(config.seed, 19)

            self.assertFalse(normal_output_root.exists())
            self.assertFalse(tuning_output_root.exists())

    def test_tuning_explicit_base_and_grid_overrides_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_root = Path(temp_dir)
            custom_base = fixture_root / "custom-base.yaml"
            custom_grid = fixture_root / "custom-grid.yaml"
            output_root = fixture_root / "outputs"
            OmegaConf.save(
                config=OmegaConf.create(
                    {
                        "override_sentinel": "custom-base",
                        "data": {"data_path": "template-data.pkl"},
                        "model": {"prediction_step": 99},
                        "saving_dir": "template-output",
                    }
                ),
                f=custom_base,
            )
            OmegaConf.save(
                config=OmegaConf.create(
                    {
                        "grid": {"model.past_window": [7]},
                        "tuning": {
                            "model_selection_valid_ratio": 0.25,
                        },
                    }
                ),
                f=custom_grid,
            )

            output = io.StringIO()
            with (
                patch.object(
                    run_kowcpi_tuning_job,
                    "REPO_ROOT",
                    fixture_root,
                ),
                contextlib.redirect_stdout(output),
            ):
                run_kowcpi_tuning_job.main(
                    [
                        "solar",
                        "--task-id",
                        "0",
                        "--base-config",
                        str(custom_base),
                        "--grid-config",
                        str(custom_grid),
                        "--output-root",
                        str(output_root),
                        "--dry-run",
                    ]
                )

            output_text = output.getvalue()
            self.assertIn(
                f"Base template: {custom_base.resolve()}",
                output_text,
            )
            self.assertIn(
                f"Tuning grid: {custom_grid.resolve()}",
                output_text,
            )
            config = _tuning_dry_run_config(output_text)
            self.assertEqual(config.override_sentinel, "custom-base")
            self.assertEqual(
                Path(config.data.data_path),
                fixture_root
                / "data"
                / "solar_prediction"
                / "lr"
                / "lr_nsdb-60m_data.pkl",
            )
            self.assertEqual(
                Path(config.saving_dir),
                fixture_root / "results" / "kowcpi" / "solar" / "lr",
            )
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
