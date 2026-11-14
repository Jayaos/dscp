"""QR-CP launchers must preserve the configured quantile head by default."""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_qr_cp import run_qr_cp_job as qr_job


REPO_ROOT = Path(__file__).resolve().parents[1]
BASH = (
    str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
    if os.name == "nt"
    else shutil.which("bash")
)


class QRCPHeadRoutingTests(unittest.TestCase):
    def preview(self, args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), patch.object(Path, "mkdir") as mkdir, patch.object(OmegaConf, "save") as save:
            qr_job.main([*args, "--dry-run"])
        mkdir.assert_not_called()
        save.assert_not_called()
        return OmegaConf.create("\n".join(stream.getvalue().splitlines()[2:]))

    def test_saved_lstm_cpu_configs_preserve_qr_templates(self):
        for dataset in qr_job.DATASET_ARTIFACTS:
            ctssf = OmegaConf.load(REPO_ROOT / "configs" / "ct_ssf_configs" / f"ct_ssf_lstm_{dataset}_config.yaml")
            threads = 10 if dataset == "solar" else 12
            for task_id in (1, 4):
                with self.subTest(dataset=dataset, task=task_id):
                    args = [dataset, "--task-id", str(task_id)]
                    original = self.preview(args)
                    output_root = REPO_ROOT / "results" / "qr_cp_lstm_saved_cpu"
                    for data_path in (ctssf.data.data_path, str((REPO_ROOT / ctssf.data.data_path).resolve())):
                        config = self.preview([
                            *args, "--data-path", data_path, "--device", "cpu",
                            "--num-threads", str(threads), "--output-root", str(output_root),
                        ])
                        self.assertEqual(Path(config.data.data_path), (REPO_ROOT / ctssf.data.data_path).resolve())
                        self.assertEqual(config.device, "cpu")
                        self.assertEqual(config.num_threads, threads)
                        self.assertEqual(config.base_predictor, "lstm")
                        self.assertEqual(config.seed, original.seed)
                        for section in ("model", "training", "plotting"):
                            self.assertEqual(config[section], original[section])
                        self.assertEqual(
                            {key: value for key, value in config.data.items() if key != "data_path"},
                            {key: value for key, value in original.data.items() if key != "data_path"},
                        )
                        encoder, _ = qr_job.TASKS[task_id]
                        self.assertEqual(Path(config.saving_dir).parent, output_root / dataset / "lstm" / encoder / config.model.head_type)

    def test_invalid_thread_count_rejected_before_config_loading(self):
        for threads in ("0", "-1", "1.5", "nan", "bad"):
            with self.subTest(threads=threads), patch.object(OmegaConf, "load") as load, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                qr_job.main(["air", "--task-id", "1", "--num-threads", threads, "--dry-run"])
            load.assert_not_called()

    def test_cpu_dispatch_applies_threads_and_uses_selected_forecast_file(self):
        for task_id in (1, 4):
            encoder, _ = qr_job.TASKS[task_id]
            config = OmegaConf.create({
                "model": {"head_type": "independent"}, "data": {"data_path": "unused.pkl"},
                "saving_dir": f"results/qr_cp/air/lstm/{encoder}/independent/fixture-run",
            })
            numpy_stub, torch_stub = types.ModuleType("numpy"), types.ModuleType("torch")
            numpy_stub.random = types.SimpleNamespace(seed=Mock())
            torch_stub.set_num_threads, torch_stub.manual_seed = Mock(), Mock()
            torch_stub.cuda = types.SimpleNamespace(is_available=Mock(), manual_seed_all=Mock())
            runner_stub = types.ModuleType("dscp.run_qr_cp")
            runner_stub.run_rnn_quantile_regression_cp = Mock()
            runner_stub.run_transformer_quantile_regression_cp = Mock()
            data_path = REPO_ROOT / "data" / "saved forecast with spaces.pkl"
            with (
                patch.dict(sys.modules, {"numpy": numpy_stub, "torch": torch_stub, "dscp.run_qr_cp": runner_stub}),
                patch.object(OmegaConf, "load", return_value=config),
                patch.object(OmegaConf, "save") as save,
                patch.object(Path, "is_file", autospec=True, return_value=True) as is_file,
                patch.object(Path, "mkdir"), contextlib.redirect_stdout(io.StringIO()),
            ):
                qr_job.main(["air", "--task-id", str(task_id), "--device", "cpu", "--num-threads", "12", "--data-path", str(data_path)])
            is_file.assert_called_once_with(data_path)
            torch_stub.set_num_threads.assert_called_once_with(12)
            torch_stub.cuda.is_available.assert_not_called()
            self.assertEqual(save.call_args.kwargs["config"].device, "cpu")
            self.assertEqual(Path(save.call_args.kwargs["config"].data.data_path), data_path)
            runner = runner_stub.run_rnn_quantile_regression_cp if encoder == "rnn" else runner_stub.run_transformer_quantile_regression_cp
            runner.assert_called_once_with(str(save.call_args.kwargs["f"]))

    def test_config_head_is_used_unless_explicitly_overridden(self):
        for task_id in (0, 3):
            encoder, predictor = qr_job.TASKS[task_id]
            for configured_head in (None, "independent", "nondecreasing"):
                for override in (None, "independent", "nondecreasing"):
                    with self.subTest(
                        encoder=encoder, configured_head=configured_head, override=override,
                    ):
                        config = OmegaConf.create({
                            "model": {"prediction_step": 1},
                            "data": {"data_path": "unused.pkl"},
                            "saving_dir": (
                                "./results/qr_cp/air/${base_predictor}/"
                                f"{encoder}/${{model.head_type}}/fixture-run"
                            ),
                        })
                        if configured_head is not None:
                            config.model.head_type = configured_head
                        expected_head = override or configured_head or "nondecreasing"
                        args = ["air", "--task-id", str(task_id), "--dry-run"]
                        if override is not None:
                            args.extend(["--head-type", override])
                        stream = io.StringIO()
                        with (
                            patch.object(OmegaConf, "load", return_value=config),
                            patch.object(Path, "mkdir") as mkdir,
                            patch.object(OmegaConf, "save") as save,
                            contextlib.redirect_stdout(stream),
                        ):
                            qr_job.main(args)

                        mkdir.assert_not_called()
                        save.assert_not_called()
                        lines = stream.getvalue().splitlines()
                        self.assertIn(f"head_type={expected_head}", lines[0])
                        resolved = OmegaConf.create("\n".join(lines[2:]))
                        self.assertEqual(resolved.model.head_type, expected_head)
                        self.assertEqual(
                            Path(resolved.saving_dir),
                            REPO_ROOT / "results" / "qr_cp" / "air" / predictor
                            / encoder / expected_head / "fixture-run",
                        )

    @unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required")
    def test_saved_lstm_cpu_sbatch_matches_resources_and_forwards_options(self):
        for dataset, (directory, artifact_name) in qr_job.DATASET_ARTIFACTS.items():
            script_path = REPO_ROOT / "sbatch" / "sbatch_run_qr_cp" / f"run_qr_cp_lstm_saved_{dataset}_cpu.sbatch"
            script = script_path.read_text()
            local_script = (REPO_ROOT / "sbatch" / "sbatch_run_local_cp" / f"run_local_cp_lstm_saved_{dataset}_cpu.sbatch").read_text()
            for line in local_script.splitlines():
                if line.startswith("#SBATCH") and not line.startswith(("#SBATCH -J ", "#SBATCH -o ")):
                    self.assertIn(line, script)
            self.assertNotIn("--gres", script)
            threads = "10" if dataset == "solar" else "12"
            for task_id in (1, 4, 0):
                for head in (None, "", "independent", "nondecreasing"):
                    with self.subTest(dataset=dataset, task=task_id, head=head):
                        env = dict(os.environ)
                        for name in ("QR_HEAD_TYPE", "QR_OUTPUT_ROOT", "QR_SEED"):
                            env.pop(name, None)
                        env.update(RUNPATH=REPO_ROOT.as_posix(), SLURM_ARRAY_TASK_ID=str(task_id), SLURM_CPUS_PER_TASK=threads)
                        if head is not None:
                            env["QR_HEAD_TYPE"] = head
                        if head:
                            env.update(QR_OUTPUT_ROOT="results/custom run with spaces", QR_SEED="47")
                        stubs = (
                            "module() { :; }\nconda() { :; }\n"
                            "python() { printf 'THREADS:%s,%s;CUDA:%s\\n' "
                            '"$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$CUDA_VISIBLE_DEVICES" >&2; '
                            "printf '%s\\0' \"$@\"; }\n"
                        )
                        result = subprocess.run([BASH, "--noprofile", "--norc", "-s"], input=stubs + script, text=True, capture_output=True, env=env, timeout=10)
                        if task_id == 0:
                            self.assertEqual(result.returncode, 2)
                            self.assertEqual(result.stdout, "")
                            continue
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stderr.strip(), f"THREADS:{threads},{threads};CUDA:")
                        args = result.stdout.rstrip("\0").split("\0")
                        self.assertEqual(args[:4], ["-u", "-m", "sbatch_run_qr_cp.run_qr_cp_job", dataset])
                        for flag, value in (("--task-id", str(task_id)), ("--device", "cpu"), ("--num-threads", threads), ("--seed", "47" if head else "2026")):
                            self.assertEqual(args[args.index(flag) + 1], value)
                        expected_data = REPO_ROOT / "data" / directory / "lstm_model_saved" / f"lstm_{artifact_name}_data.pkl"
                        self.assertEqual(Path(args[args.index("--data-path") + 1]), expected_data)
                        expected_output = Path("results/custom run with spaces") if head else REPO_ROOT / "results" / "qr_cp_lstm_saved_cpu"
                        self.assertEqual(Path(args[args.index("--output-root") + 1]), expected_output)
                        if head:
                            self.assertEqual(args[args.index("--head-type") + 1], head)
                        else:
                            self.assertNotIn("--head-type", args)

    @unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required")
    def test_sbatch_only_overrides_head_when_environment_value_is_set(self):
        for dataset in qr_job.DATASET_ARTIFACTS:
            script = (
                REPO_ROOT / "sbatch" / "sbatch_run_qr_cp" / f"run_qr_cp_{dataset}.sbatch"
            ).read_text(encoding="utf-8")
            for head in (None, "", "independent", "nondecreasing"):
                with self.subTest(dataset=dataset, head=head):
                    env = dict(os.environ)
                    env.pop("QR_HEAD_TYPE", None)
                    env.update({
                        "RUNPATH": REPO_ROOT.as_posix(),
                        "SLURM_CPUS_PER_TASK": "1",
                        "SLURM_ARRAY_TASK_ID": "3",
                    })
                    if head is not None:
                        env["QR_HEAD_TYPE"] = head
                    # Capture the actual arguments without loading an environment
                    # or starting training on this machine.
                    stubs = (
                        "module() { :; }\n"
                        "conda() { :; }\n"
                        "python() { printf '%s\\0' \"$@\"; }\n"
                    )
                    result = subprocess.run(
                        [BASH, "--noprofile", "--norc", "-s"],
                        input=stubs + script,
                        text=True,
                        capture_output=True,
                        env=env,
                        timeout=10,
                        check=True,
                    )
                    arguments = result.stdout.rstrip("\0").split("\0")
                    self.assertEqual(arguments[:4], [
                        "-u", "-m", "sbatch_run_qr_cp.run_qr_cp_job", dataset,
                    ])
                    if head:
                        self.assertEqual(arguments.count("--head-type"), 1)
                        self.assertEqual(arguments[arguments.index("--head-type") + 1], head)
                    else:
                        self.assertNotIn("--head-type", arguments)


if __name__ == "__main__":
    unittest.main()
