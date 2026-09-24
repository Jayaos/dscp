"""QR-CP launchers must preserve the configured quantile head by default."""

import contextlib
import io
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_qr_cp import run_qr_cp_job as qr_job


REPO_ROOT = Path(__file__).resolve().parents[1]
BASH = (
    str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
    if os.name == "nt"
    else shutil.which("bash")
)


class QRCPHeadRoutingTests(unittest.TestCase):
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
