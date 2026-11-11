"""Local-CP kernel routing and opt-in Slurm launchers."""

import contextlib
import io
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_local_cp import run_local_cp_job as local_job


REPO_ROOT = Path(__file__).resolve().parents[1]
BASH = (
    str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
    if os.name == "nt"
    else shutil.which("bash")
)


def fixture_config(dataset="air", encoder="rnn"):
    return OmegaConf.create({
        "model": {
            "prediction_step": 1,
            "similarity_fn": "dot_product",
            "temperature": 7.0,
            "sampling_num": 1234,
            "rolling_calibration": False,
        },
        "data": {"data_path": "unused.pkl"},
        "saving_dir": (
            f"./results/local_cp/{dataset}/${{base_predictor}}/{encoder}/fixture-run"
        ),
    })


class LocalCPKernelRoutingTests(unittest.TestCase):
    def preview(self, args, config=None, expected_template=None):
        stream = io.StringIO()
        with (
            patch.object(OmegaConf, "load", return_value=config)
            if config is not None else contextlib.nullcontext(),
            patch.object(Path, "mkdir") as mkdir,
            patch.object(OmegaConf, "save") as save,
            contextlib.redirect_stdout(stream),
        ):
            local_job.main([*args, "--dry-run"])
        mkdir.assert_not_called()
        save.assert_not_called()
        lines = stream.getvalue().splitlines()
        if expected_template is not None:
            self.assertEqual(lines[1], f"Configuration template: {expected_template}")
        return lines[0], OmegaConf.create("\n".join(lines[2:]))

    def test_default_softmax_preserves_weighting_and_output(self):
        for task_id in (0, 3):
            encoder, predictor = local_job.TASKS[task_id]
            with self.subTest(encoder=encoder):
                header, config = self.preview(
                    ["air", "--task-id", str(task_id)], fixture_config(encoder=encoder),
                    expected_template=REPO_ROOT / "configs" / "lcp_configs"
                    / f"lcp_{encoder}_{predictor}_air_config.yaml",
                )
                self.assertIn("kernel=softmax", header)
                self.assertEqual(config.model.kernel, "softmax")
                self.assertEqual(config.model.similarity_fn, "dot_product")
                self.assertEqual(config.model.temperature, 7.0)
                self.assertNotIn("bandwidth", config.model)
                self.assertEqual(
                    Path(config.saving_dir), REPO_ROOT / "results" / "local_cp"
                    / "air" / predictor / encoder / "fixture-run",
                )

    def test_epanechnikov_routes_all_tasks_and_preserves_other_settings(self):
        for dataset, (artifact_dir, artifact_name) in local_job.DATASET_ARTIFACTS.items():
            for task_id, (encoder, predictor) in enumerate(local_job.TASKS):
                with self.subTest(dataset=dataset, task_id=task_id):
                    header, config = self.preview([
                        dataset, "--task-id", str(task_id), "--kernel", "epanechnikov",
                        "--bandwidth", "0.7500001", "--seed", "47",
                    ], fixture_config(dataset, encoder),
                        expected_template=REPO_ROOT / "configs" / "lcp_epanechnikov_kernel_configs"
                        / f"lcp_{encoder}_epanechnikov_{predictor}_{dataset}_config.yaml",
                    )
                    self.assertIn("kernel=epanechnikov", header)
                    self.assertEqual(config.model.kernel, "epanechnikov")
                    self.assertEqual(config.model.bandwidth, 0.7500001)
                    self.assertEqual(config.model.similarity_fn, "cos_similarity")
                    self.assertEqual(config.model.temperature, 7.0)
                    self.assertEqual(config.model.sampling_num, 1234)
                    self.assertFalse(config.model.rolling_calibration)
                    self.assertEqual(config.seed, 47)
                    self.assertEqual(config.base_predictor, predictor)
                    self.assertEqual(
                        Path(config.data.data_path), REPO_ROOT / "data" / artifact_dir
                        / predictor / f"{predictor}_{artifact_name}_data.pkl",
                    )
                    self.assertEqual(
                        Path(config.saving_dir), REPO_ROOT / "results" / "local_cp_epanechnikov"
                        / "bandwidth_0.7500001" / "seed_47" / dataset / predictor
                        / encoder / "fixture-run",
                    )

    def test_epanechnikov_uses_selected_bandwidths_from_real_templates(self):
        # Each row follows TASKS: RNN LR/LSTM/Chronos, then Transformer LR/LSTM/Chronos.
        bandwidths = {
            "air": (0.75, 1.0, 0.75, 0.75, 1.0, 0.75),
            "solar": (1.0, 0.75, 0.75, 0.75, 0.75, 0.75),
            "sapflux": (0.75, 0.75, 0.5, 0.5, 0.5, 0.25),
        }
        for dataset, expected_bandwidths in bandwidths.items():
            for task_id, (encoder, predictor) in enumerate(local_job.TASKS):
                with self.subTest(dataset=dataset, task_id=task_id):
                    _, config = self.preview(
                        [dataset, "--task-id", str(task_id), "--kernel", "epanechnikov"],
                        expected_template=REPO_ROOT / "configs" / "lcp_epanechnikov_kernel_configs"
                        / f"lcp_{encoder}_epanechnikov_{predictor}_{dataset}_config.yaml",
                    )
                    bandwidth = expected_bandwidths[task_id]
                    self.assertEqual(config.model.kernel, "epanechnikov")
                    self.assertEqual(config.model.bandwidth, bandwidth)
                    self.assertEqual(config.model.similarity_fn, "cos_similarity")
                    self.assertEqual(config.base_predictor, predictor)
                    self.assertEqual(
                        Path(config.saving_dir).parent,
                        REPO_ROOT / "results" / "local_cp_epanechnikov"
                        / f"bandwidth_{bandwidth}" / "seed_2026" / dataset / predictor / encoder,
                    )

    def test_template_kernel_bandwidth_and_explicit_overrides(self):
        config = fixture_config()
        config.model.kernel = "epanechnikov"
        config.model.bandwidth = 0.5
        _, resolved = self.preview(["air", "--task-id", "0"], config)
        self.assertEqual(resolved.model.bandwidth, 0.5)
        self.assertEqual(resolved.model.kernel, "epanechnikov")

        config = fixture_config()
        config.model.kernel = "epanechnikov"
        config.model.bandwidth = 0.5
        output_root = REPO_ROOT / "results" / "custom experiment"
        _, resolved = self.preview([
            "air", "--task-id", "0", "--kernel", "epanechnikov", "--bandwidth", "1.25",
            "--output-root", str(output_root),
        ], config)
        self.assertEqual(resolved.model.bandwidth, 1.25)
        self.assertEqual(Path(resolved.saving_dir), output_root / "air" / "lr" / "rnn" / "fixture-run")

        config = fixture_config()
        config.model.kernel = "epanechnikov"
        _, resolved = self.preview(["air", "--task-id", "0", "--kernel", "softmax"], config)
        self.assertEqual(resolved.model.kernel, "softmax")
        self.assertEqual(resolved.model.similarity_fn, "dot_product")

    def test_invalid_bandwidths_and_missing_bandwidth_fail_before_execution(self):
        cases = [
            (["--kernel", "epanechnikov"], "requires --bandwidth"),
            (["--bandwidth", "1.0"], "applies only"),
        ]
        cases.extend((
            ["--kernel", "epanechnikov", "--bandwidth", bandwidth], "finite and positive",
        ) for bandwidth in ("0", "-1", "nan", "inf", "1e309", "bad"))
        for options, message in cases:
            with self.subTest(options=options):
                stream = io.StringIO()
                with (
                    patch.object(OmegaConf, "load", return_value=fixture_config()),
                    patch.object(Path, "mkdir") as mkdir,
                    contextlib.redirect_stderr(stream),
                    self.assertRaises(SystemExit) as raised,
                ):
                    local_job.main(["air", "--task-id", "0", *options, "--dry-run"])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stream.getvalue())
                mkdir.assert_not_called()

        for bandwidth in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(config_bandwidth=bandwidth):
                config = fixture_config()
                config.model.kernel = "epanechnikov"
                config.model.bandwidth = bandwidth
                with (
                    patch.object(OmegaConf, "load", return_value=config),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    local_job.main(["air", "--task-id", "0", "--dry-run"])

    @unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required")
    def test_epanechnikov_arrays_forward_bandwidth_seed_and_optional_output(self):
        for dataset in local_job.DATASET_ARTIFACTS:
            for output_root in (None, "results/experiment with spaces"):
                with self.subTest(dataset=dataset, output_root=output_root):
                    env = self.array_environment()
                    env.update({"LOCAL_CP_BANDWIDTH": "0.75", "LOCAL_CP_SEED": "47"})
                    if output_root is not None:
                        env["LOCAL_CP_OUTPUT_ROOT"] = output_root
                    result = self.run_array(dataset, env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    arguments = result.stdout.rstrip("\0").split("\0")
                    self.assertEqual(arguments[:4], [
                        "-u", "-m", "sbatch_run_local_cp.run_local_cp_job", dataset,
                    ])
                    for flag, value in (("--task-id", "3"), ("--kernel", "epanechnikov"),
                                        ("--bandwidth", "0.75"), ("--seed", "47")):
                        self.assertEqual(arguments.count(flag), 1)
                        self.assertEqual(arguments[arguments.index(flag) + 1], value)
                    if output_root is None:
                        self.assertNotIn("--output-root", arguments)
                    else:
                        self.assertEqual(arguments[arguments.index("--output-root") + 1], output_root)

    @unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required")
    def test_epanechnikov_arrays_default_to_template_bandwidth(self):
        for dataset in local_job.DATASET_ARTIFACTS:
            for bandwidth in (None, ""):
                with self.subTest(dataset=dataset, bandwidth=bandwidth):
                    env = self.array_environment()
                    if bandwidth is not None:
                        env["LOCAL_CP_BANDWIDTH"] = bandwidth
                    result = self.run_array(dataset, env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    arguments = result.stdout.rstrip("\0").split("\0")
                    self.assertEqual(arguments[:4], [
                        "-u", "-m", "sbatch_run_local_cp.run_local_cp_job", dataset,
                    ])
                    self.assertNotIn("--bandwidth", arguments)
                    for flag, value in (("--task-id", "3"), ("--kernel", "epanechnikov"),
                                        ("--seed", "2026")):
                        self.assertEqual(arguments.count(flag), 1)
                        self.assertEqual(arguments[arguments.index(flag) + 1], value)

    @staticmethod
    def array_environment():
        env = dict(os.environ)
        for key in ("LOCAL_CP_BANDWIDTH", "LOCAL_CP_OUTPUT_ROOT", "LOCAL_CP_SEED"):
            env.pop(key, None)
        env.update({
            "RUNPATH": REPO_ROOT.as_posix(),
            "SLURM_CPUS_PER_TASK": "1",
            "SLURM_ARRAY_TASK_ID": "3",
        })
        return env

    @staticmethod
    def run_array(dataset, env):
        script = (
            REPO_ROOT / "sbatch" / "sbatch_run_local_cp"
            / f"run_local_cp_epanechnikov_{dataset}.sbatch"
        ).read_text(encoding="utf-8")
        stubs = (
            "module() { :; }\n"
            "conda() { :; }\n"
            "python() { printf '%s\\0' \"$@\"; }\n"
        )
        return subprocess.run(
            [BASH, "--noprofile", "--norc", "-s"], input=stubs + script,
            text=True, capture_output=True, env=env, timeout=10,
        )


if __name__ == "__main__":
    unittest.main()
