"""Exercise Solar's allocation-sized launch orchestration without Slurm."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from sbatch.sbatch_run_distmatch import run_distmatch as cli


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "sbatch/sbatch_run_distmatch/run_distmatch_solar.sbatch"
END_RECORD = "__END_DISTMATCH_TEST__"


class DistMatchSolarLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        cls.bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if cls.bash is None:
            raise unittest.SkipTest("Bash is required to exercise the Solar launcher")
        try:
            probe = subprocess.run(
                [cls.bash, "--version"], capture_output=True, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise unittest.SkipTest(f"Bash is unavailable: {exc}") from exc
        if probe.returncode:
            raise unittest.SkipTest("The available Bash executable cannot run")

    def _launch(
        self, *arguments, cpus=17, task=0, fail_stage="", fail_rank="",
        job_nodes=3, nnodes=3,
    ):
        with tempfile.TemporaryDirectory(prefix="distmatch-solar-launch-") as directory:
            root = Path(directory)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            conda_hook = root / "conda/etc/profile.d/conda.sh"
            conda_hook.parent.mkdir(parents=True)
            conda_hook.write_text("# The mock environment needs no activation.\n", encoding="utf-8")

            # An executable, rather than a shell function, also intercepts
            # ``exec python`` inside the Bash command launched for each rank.
            mock_python = mock_bin / "python"
            mock_python.write_bytes((
                "#!/bin/bash\n"
                "printf '\\0__PYTHON__\\0%s\\0' \"${SLURM_PROCID:-batch}\"\n"
                "printf '%s\\0' \"$@\"\n"
                f"printf '{END_RECORD}\\0'\n"
                "for argument in \"$@\"; do\n"
                "  if [[ \"$argument\" == --prepare-shards && \"${MOCK_FAIL_STAGE:-}\" == prepare ]]; then\n"
                "    exit 41\n"
                "  fi\n"
                "  if [[ \"$argument\" == --shard-index && \"${SLURM_PROCID:-}\" == \"${MOCK_FAIL_RANK:-never}\" ]]; then\n"
                "    exit 42\n"
                "  fi\n"
                "done\n"
            ).encode("utf-8"))
            mock_python.chmod(0o755)
            environment = os.environ.copy()
            for name in list(environment):
                if name.startswith(("SLURM_", "DISTMATCH_")) or name in {
                    "BASH_ENV", "ENV", "RUNPATH", "DSCP_RUNPATH",
                }:
                    environment.pop(name)
            environment.update(
                PATH=str(mock_bin) + os.pathsep + environment.get("PATH", ""),
                DSCP_RUNPATH=REPO_ROOT.as_posix(),
                MOCK_CONDA_ROOT=(root / "conda").as_posix(),
                MOCK_FAIL_STAGE=fail_stage,
                MOCK_FAIL_RANK=str(fail_rank),
                SLURM_ARRAY_JOB_ID="4815",
                SLURM_JOB_ID="4816",
                SLURM_ARRAY_TASK_ID=str(task),
                SLURM_NTASKS=str(job_nodes or nnodes or 3),
                SLURM_CPUS_PER_TASK=str(cpus),
            )
            if job_nodes is not None:
                environment["SLURM_JOB_NUM_NODES"] = str(job_nodes)
            if nnodes is not None:
                environment["SLURM_NNODES"] = str(nnodes)
            harness = (
                "module() { :; }\n"
                "conda() {\n"
                "  if [[ \"${1:-}\" == info && \"${2:-}\" == --base ]]; then\n"
                "    printf '%s\\n' \"$MOCK_CONDA_ROOT\"\n"
                "  fi\n"
                "}\n"
                "srun() {\n"
                "  printf '\\0__SRUN__\\0'\n"
                "  printf '%s\\0' \"$@\"\n"
                f"  printf '{END_RECORD}\\0'\n"
                "  local task_count='' argument\n"
                "  for argument in \"$@\"; do\n"
                "    case \"$argument\" in --ntasks=*) task_count=\"${argument#--ntasks=}\" ;; esac\n"
                "  done\n"
                "  [[ \"$task_count\" =~ ^[1-9][0-9]*$ ]] || return 91\n"
                "  while (( $# )) && [[ \"$1\" != bash ]]; do shift; done\n"
                "  (( $# )) || return 90\n"
                "  shift\n"
                "  local rank\n"
                "  for ((rank = 0; rank < task_count; rank++)); do\n"
                "    SLURM_PROCID=\"$rank\" \"$BASH\" \"$@\" || return $?\n"
                "  done\n"
                "}\n"
            )
            # read_text normalizes CRLF in Windows checkouts before Bash reads it.
            result = subprocess.run(
                [self.bash, "--noprofile", "--norc", "-s", "--", *arguments],
                input=(harness + SCRIPT.read_text(encoding="utf-8")).encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=REPO_ROOT, env=environment, timeout=20, check=False,
            )
        return result, self._records(result.stdout)

    @staticmethod
    def _records(output):
        tokens = iter(output.decode("utf-8").split("\0"))
        records = []
        for token in tokens:
            if token not in {"__PYTHON__", "__SRUN__"}:
                continue
            payload = []
            for value in tokens:
                if value == END_RECORD:
                    break
                payload.append(value)
            else:
                raise AssertionError("Incomplete mocked command record")
            records.append((token, payload))
        return records

    def _python_calls(self, records):
        calls = []
        for kind, payload in records:
            if kind != "__PYTHON__":
                continue
            rank, *arguments = payload
            self.assertEqual(arguments[:3], ["-u", "-m", "sbatch.sbatch_run_distmatch.run_distmatch"])
            calls.append((rank, cli.build_parser().parse_args(arguments[3:])))
        return calls

    def test_resource_request_and_bash_syntax(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --array=0-1", "#SBATCH --nodes=3", "#SBATCH --ntasks=3",
            "#SBATCH --ntasks-per-node=1", "#SBATCH --cpus-per-task=17",
        ):
            self.assertIn(directive, source)
        result = subprocess.run(
            [self.bash, "--noprofile", "--norc", "-n"], input=source.encode("utf-8"),
            capture_output=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))

    def test_two_predictors_prepare_all_three_ranks_then_merge_in_separate_directories(self):
        outputs = []
        for task, predictor in ((0, "lr"), (1, "lstm")):
            with self.subTest(predictor=predictor):
                result, records = self._launch(task=task)
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
                self.assertEqual([kind for kind, _ in records], [
                    "__PYTHON__", "__SRUN__", "__PYTHON__", "__PYTHON__",
                    "__PYTHON__", "__PYTHON__",
                ])
                launch_arguments = records[1][1]
                for option in (
                    "--nodes=3", "--ntasks=3", "--ntasks-per-node=1",
                    "--cpus-per-task=17", "--kill-on-bad-exit=1", "--export=ALL",
                ):
                    self.assertIn(option, launch_arguments)
                calls = self._python_calls(records)
                self.assertEqual(
                    [rank for rank, _ in calls], ["batch", "0", "1", "2", "batch"],
                )
                self.assertEqual(calls[0][1].prepare_shards, 3)
                self.assertEqual([call.shard_index for _, call in calls[1:4]], [0, 1, 2])
                self.assertTrue(calls[-1][1].merge_shards)
                for _, call in calls:
                    self.assertEqual(call.num_cores, 17)
                    self.assertEqual(call.config_path.name, f"distmatch_{predictor}_solar_config.yaml")
                    self.assertEqual(call.output_dir, calls[0][1].output_dir)
                output = calls[0][1].output_dir.as_posix()
                self.assertTrue(output.endswith(f"results/distmatch_solar/job_4815/{predictor}"), output)
                outputs.append(output)
        self.assertNotEqual(outputs[0], outputs[1])

    def test_failed_preparation_or_rank_never_merges(self):
        for fail_stage, fail_rank, expected_ranks in (
            ("prepare", "", ["batch"]),
            ("", "0", ["batch", "0"]),
            ("", "1", ["batch", "0", "1"]),
            ("", "2", ["batch", "0", "1", "2"]),
        ):
            with self.subTest(fail_stage=fail_stage, fail_rank=fail_rank):
                result, records = self._launch(fail_stage=fail_stage, fail_rank=fail_rank)
                self.assertNotEqual(result.returncode, 0)
                calls = self._python_calls(records)
                self.assertEqual([rank for rank, _ in calls], expected_ranks)
                self.assertFalse(any(call.merge_shards for _, call in calls))

    def test_worker_count_uses_allocation_and_accepts_explicit_override(self):
        for cpus, arguments in ((16, ()), (17, ("--num-cores", "16"))):
            with self.subTest(cpus=cpus, arguments=arguments):
                result, records = self._launch(*arguments, cpus=cpus)
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
                calls = self._python_calls(records)
                self.assertEqual(len(calls), 5)
                self.assertTrue(all(call.num_cores == 16 for _, call in calls))

    def test_shard_count_follows_slurm_node_count_with_documented_fallback(self):
        for job_nodes, nnodes, expected in ((4, 2, 4), (None, 2, 2), (None, None, 3)):
            with self.subTest(job_nodes=job_nodes, nnodes=nnodes):
                result, records = self._launch(job_nodes=job_nodes, nnodes=nnodes)
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
                launch_arguments = next(payload for kind, payload in records if kind == "__SRUN__")
                self.assertIn(f"--nodes={expected}", launch_arguments)
                self.assertIn(f"--ntasks={expected}", launch_arguments)
                calls = self._python_calls(records)
                self.assertEqual(calls[0][1].prepare_shards, expected)
                self.assertEqual(
                    [call.shard_index for _, call in calls[1:-1]], list(range(expected)),
                )
                self.assertTrue(calls[-1][1].merge_shards)

    def test_invalid_slurm_node_count_fails_before_preparation(self):
        for job_nodes in (0, -1, "three", "2.5"):
            with self.subTest(job_nodes=job_nodes):
                result, records = self._launch(job_nodes=job_nodes, nnodes=2)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(records, [])
                self.assertIn(
                    "Expected a positive Slurm node count",
                    result.stderr.decode("utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
