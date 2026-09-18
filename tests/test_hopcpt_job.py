"""Check HopCPT job configuration and CUDA allocation before launching training."""

import contextlib
import importlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_job_module():
    sbatch_path = str(REPO_ROOT / "sbatch")
    with patch.object(sys, "path", [sbatch_path, *sys.path]):
        return importlib.import_module("sbatch_run_hopcpt.run_hopcpt_job")


def _training_stubs(*, available=True, device_count=4):
    """Keep dispatcher tests independent of Torch, NumPy, and model dependencies."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=Mock(return_value=available),
        device_count=Mock(return_value=device_count),
        manual_seed_all=Mock(),
    )
    torch.manual_seed = Mock()
    torch.set_num_threads = Mock()
    numpy = types.ModuleType("numpy")
    numpy.random = types.SimpleNamespace(seed=Mock())
    runner = types.ModuleType("baselines.hopcpt.run_hopcpt")
    runner.run_hopcpt = Mock()
    return {
        "torch": torch,
        "numpy": numpy,
        "baselines.hopcpt.run_hopcpt": runner,
    }


class HopCPTJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = _load_job_module()

    def _dry_run(self, arguments, *, cpu_budget=None):
        environment = {} if cpu_budget is None else {
            "SLURM_CPUS_PER_TASK": str(cpu_budget)
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.dict(
                sys.modules,
                {
                    "torch": None,
                    "numpy": None,
                    "baselines.hopcpt.run_hopcpt": None,
                },
            ),
            patch.object(Path, "is_file", return_value=False),
            patch.object(Path, "mkdir") as mkdir,
            patch.object(OmegaConf, "save") as save_config,
            patch.object(OmegaConf, "to_yaml", wraps=OmegaConf.to_yaml) as to_yaml,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.job.main([*arguments, "--dry-run"])

        mkdir.assert_not_called()
        save_config.assert_not_called()
        to_yaml.assert_called_once()
        return to_yaml.call_args.args[0]

    def test_all_dry_runs_resolve_without_artifacts_cuda_or_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            for dataset in self.job.DATASET_ARTIFACTS:
                for task_id, predictor in enumerate(self.job.TASKS):
                    for num_gpus in (1, 2, 4):
                        with self.subTest(
                            dataset=dataset, predictor=predictor, num_gpus=num_gpus
                        ):
                            config = self._dry_run(
                                [
                                    dataset,
                                    "--task-id", str(task_id),
                                    "--num-gpus", str(num_gpus),
                                    "--output-root", str(output_root),
                                    "--seed", "17",
                                ],
                                cpu_budget=8,
                            )
                            self.assertEqual(config.parallel.enabled, num_gpus > 1)
                            self.assertEqual(
                                list(config.parallel.devices), list(range(num_gpus))
                            )
                            self.assertEqual(
                                config.parallel.threads_per_worker, 8 // num_gpus
                            )
                            self.assertEqual(config.device, 0)
                            self.assertEqual(config.seed, 17)
                            self.assertEqual(config.model.prediction_step, 1)
                            artifact_dir, artifact_name = (
                                self.job.DATASET_ARTIFACTS[dataset]
                            )
                            self.assertEqual(
                                Path(config.data.data_path),
                                REPO_ROOT / "data" / artifact_dir / predictor
                                / f"{predictor}_{artifact_name}_data.pkl",
                            )
                            self.assertEqual(
                                Path(config.saving_dir),
                                output_root.resolve() / dataset / predictor,
                            )

            self.assertFalse(output_root.exists())

    def test_thread_budget_defaults_and_small_allocations(self):
        for cpu_budget, num_gpus, expected_threads in (
            (None, 1, 1),
            (None, 4, 1),
            (2, 4, 1),
            (7, 2, 3),
        ):
            with self.subTest(cpu_budget=cpu_budget, num_gpus=num_gpus):
                config = self._dry_run(
                    ["solar", "--task-id", "0", "--num-gpus", str(num_gpus)],
                    cpu_budget=cpu_budget,
                )
                self.assertEqual(config.parallel.threads_per_worker, expected_threads)

    def test_parser_defaults_to_one_gpu_and_rejects_invalid_counts(self):
        parser = self.job.build_parser()
        self.assertEqual(parser.parse_args(["solar", "--task-id", "0"]).num_gpus, 1)
        for value in ("0", "-1", "1.5", "two"):
            with self.subTest(value=value):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    parser.parse_args(
                        ["solar", "--task-id", "0", "--num-gpus", value]
                    )
                self.assertEqual(raised.exception.code, 2)

    def test_last_cli_gpu_override_restores_single_gpu_execution(self):
        config = self._dry_run(
            ["solar", "--task-id", "0", "--num-gpus", "2", "--num-gpus", "1"]
        )
        self.assertFalse(config.parallel.enabled)
        self.assertEqual(list(config.parallel.devices), [0])

    def test_insufficient_cuda_allocation_fails_before_writing_or_training(self):
        for available, device_count, requested in (
            (False, 0, 1),
            (False, 4, 2),
            (True, 1, 2),
            (True, 2, 4),
        ):
            with self.subTest(
                available=available, device_count=device_count, requested=requested
            ):
                modules = _training_stubs(
                    available=available, device_count=device_count
                )
                stderr = io.StringIO()
                with (
                    patch.dict(sys.modules, modules),
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(Path, "is_file", return_value=True),
                    patch.object(Path, "mkdir") as mkdir,
                    patch.object(OmegaConf, "save") as save_config,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as raised,
                ):
                    self.job.main(
                        ["solar", "--task-id", "0", "--num-gpus", str(requested)]
                    )
                self.assertEqual(raised.exception.code, 2)
                self.assertRegex(stderr.getvalue(), "(?i)cuda|gpu")
                mkdir.assert_not_called()
                save_config.assert_not_called()
                modules["baselines.hopcpt.run_hopcpt"].run_hopcpt.assert_not_called()

    def test_training_receives_resolved_gpu_allocation_and_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for num_gpus in (1, 2, 4):
                with self.subTest(num_gpus=num_gpus):
                    modules = _training_stubs()
                    with (
                        patch.dict(sys.modules, modules),
                        patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}),
                        patch.object(Path, "is_file", return_value=True),
                        patch.object(Path, "mkdir") as mkdir,
                        patch.object(OmegaConf, "save") as save_config,
                        patch("random.seed") as random_seed,
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        self.job.main(
                            [
                                "sapflux", "--task-id", "2",
                                "--num-gpus", str(num_gpus),
                                "--output-root", temp_dir,
                                "--seed", "29",
                            ]
                        )

                    mkdir.assert_called_once_with(parents=True, exist_ok=True)
                    save_config.assert_called_once()
                    saved = save_config.call_args.kwargs["config"]
                    self.assertEqual(saved.parallel.enabled, num_gpus > 1)
                    self.assertEqual(list(saved.parallel.devices), list(range(num_gpus)))
                    self.assertEqual(saved.parallel.threads_per_worker, 8 // num_gpus)
                    self.assertEqual(saved.seed, 29)
                    random_seed.assert_called_once_with(29)
                    modules["numpy"].random.seed.assert_called_once_with(29)
                    modules["torch"].manual_seed.assert_called_once_with(29)
                    modules["torch"].set_num_threads.assert_called_once_with(8 // num_gpus)
                    modules["torch"].cuda.manual_seed_all.assert_called_once_with(29)
                    config_path = (
                        Path(temp_dir).resolve() / "sapflux" / "chronos"
                        / "resolved_config.yaml"
                    )
                    self.assertEqual(save_config.call_args.kwargs["f"], config_path)
                    modules["baselines.hopcpt.run_hopcpt"].run_hopcpt.assert_called_once_with(
                        str(config_path)
                    )

    def _launch_sbatch_script(self, dataset, num_gpus, *extra_args):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if bash is None:
            self.skipTest("Bash is required to exercise the Slurm launcher")

        script_path = (
            REPO_ROOT / "sbatch" / "sbatch_run_hopcpt" / f"run_hopcpt_{dataset}.sbatch"
        )
        environment = os.environ.copy()
        environment.pop("BASH_ENV", None)
        environment.pop("SLURM_GPUS_ON_NODE", None)
        environment.update(
            RUNPATH=REPO_ROOT.as_posix(),
            SLURM_CPUS_PER_TASK="4",
            SLURM_ARRAY_TASK_ID="1",
        )
        if num_gpus is not None:
            environment["SLURM_GPUS_ON_NODE"] = str(num_gpus)

        # Shell functions intercept cluster setup and capture Python arguments.
        # Reading as text also normalizes Windows checkout line endings for Bash.
        harness = (
            "module() { :; }\n"
            "conda() { :; }\n"
            "python() { printf '%s\\0' \"$@\"; }\n"
        )
        return subprocess.run(
            [bash, "--noprofile", "--norc", "-s", "--", *extra_args],
            input=(harness + script_path.read_text(encoding="utf-8")).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=REPO_ROOT,
            env=environment,
            timeout=15,
            check=False,
        )

    def test_sbatch_scripts_use_the_actual_slurm_gpu_allocation(self):
        for dataset in self.job.DATASET_ARTIFACTS:
            for num_gpus in (1, 2, 3, 10):
                with self.subTest(dataset=dataset, num_gpus=num_gpus):
                    result = self._launch_sbatch_script(dataset, num_gpus)
                    self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
                    arguments = result.stdout.decode("utf-8").split("\0")[:-1]
                    self.assertEqual(
                        arguments[:4],
                        ["-u", "-m", "sbatch_run_hopcpt.run_hopcpt_job", dataset],
                    )
                    parsed = self.job.build_parser().parse_args(arguments[3:])
                    self.assertEqual(parsed.num_gpus, num_gpus)
                    self.assertEqual(parsed.task_id, 1)

    def test_sbatch_scripts_allow_explicit_gpu_override(self):
        for dataset in self.job.DATASET_ARTIFACTS:
            with self.subTest(dataset=dataset):
                result = self._launch_sbatch_script(dataset, 2, "--num-gpus", "1")
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
                arguments = result.stdout.decode("utf-8").split("\0")[:-1]
                parsed = self.job.build_parser().parse_args(arguments[3:])
                self.assertEqual(parsed.num_gpus, 1)

    def test_sbatch_scripts_require_the_slurm_gpu_allocation_before_python(self):
        for dataset in self.job.DATASET_ARTIFACTS:
            with self.subTest(dataset=dataset):
                result = self._launch_sbatch_script(dataset, None)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
                self.assertIn(b"SLURM_GPUS_ON_NODE", result.stderr)


if __name__ == "__main__":
    unittest.main()
