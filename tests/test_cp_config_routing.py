"""QR-CP and SPCI jobs must read each task's own experiment settings."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_qr_cp import run_qr_cp_job as qr_job
from sbatch.sbatch_run_spci import run_spci_job as spci_job


REPO_ROOT = Path(__file__).resolve().parents[1]


class CPConfigRoutingTests(unittest.TestCase):
    def _dry_run(self, job, dataset, task_id, *, root=REPO_ROOT, extra_args=()):
        stream = io.StringIO()
        with (
            patch.object(job, "REPO_ROOT", root),
            patch.object(Path, "mkdir") as mkdir,
            patch.object(OmegaConf, "save") as save,
            contextlib.redirect_stdout(stream),
        ):
            job.main([dataset, "--task-id", str(task_id), "--dry-run", *extra_args])
        mkdir.assert_not_called()
        save.assert_not_called()
        lines = stream.getvalue().splitlines()
        self.assertTrue(lines[1].startswith("Configuration template: "))
        template = Path(lines[1].split(": ", 1)[1])
        return template, OmegaConf.create("\n".join(lines[2:]))

    def _assert_artifact(self, job, config, dataset, predictor, root=REPO_ROOT):
        directory, name = job.DATASET_ARTIFACTS[dataset]
        self.assertEqual(
            Path(config.data.data_path),
            root / "data" / directory / predictor / f"{predictor}_{name}_data.pkl",
        )

    def test_qr_uses_all_dedicated_configs_and_preserves_settings(self):
        for dataset in qr_job.DATASET_ARTIFACTS:
            for task_id, (encoder, predictor) in enumerate(qr_job.TASKS):
                for head in ("nondecreasing", "independent"):
                    with self.subTest(dataset=dataset, task=task_id, head=head):
                        template, config = self._dry_run(
                            qr_job, dataset, task_id, extra_args=("--head-type", head)
                        )
                        expected = (
                            REPO_ROOT / "configs" / "qr_cp_configs"
                            / f"qr_{encoder}_{predictor}_{dataset}_config.yaml"
                        )
                        self.assertEqual(template, expected)
                        source = OmegaConf.load(expected)
                        source.model.head_type = head
                        source.model.prediction_step = 1
                        self.assertEqual(config.model, source.model)
                        self.assertEqual(config.training, source.training)
                        self.assertEqual(
                            config.data.strided_features, source.data.strided_features
                        )
                        self.assertEqual(config.data.normalize, source.data.normalize)
                        self.assertEqual(config.base_predictor, predictor)
                        self._assert_artifact(qr_job, config, dataset, predictor)
                        self.assertEqual(
                            Path(config.saving_dir),
                            REPO_ROOT / "results" / "qr_cp" / dataset
                            / predictor / encoder / head,
                        )

    def test_spci_uses_all_dedicated_configs_and_preserves_settings(self):
        for dataset in spci_job.DATASET_ARTIFACTS:
            for task_id, predictor in enumerate(spci_job.TASKS):
                with self.subTest(dataset=dataset, predictor=predictor):
                    template, config = self._dry_run(spci_job, dataset, task_id)
                    expected = (
                        REPO_ROOT / "configs" / "spci_configs"
                        / f"spci_{predictor}_{dataset}_config.yaml"
                    )
                    self.assertEqual(template, expected)
                    source = OmegaConf.load(expected)
                    source.model.prediction_step = 1
                    self.assertEqual(config.model, source.model)
                    self.assertEqual(config.data.normalize, source.data.normalize)
                    self.assertEqual(config.data.train_ratio, source.data.train_ratio)
                    self._assert_artifact(spci_job, config, dataset, predictor)
                    self.assertEqual(
                        Path(config.saving_dir),
                        REPO_ROOT / "results" / "spci" / dataset / predictor,
                    )

    def test_distinct_route_settings_cannot_be_replaced_by_shared_templates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            routes = []
            for job, method, folder in (
                (qr_job, "qr", "qr_cp_configs"),
                (spci_job, "spci", "spci_configs"),
            ):
                config_dir = root / "configs" / folder
                config_dir.mkdir(parents=True)
                for dataset in job.DATASET_ARTIFACTS:
                    for task_id, task in enumerate(job.TASKS):
                        if method == "qr":
                            encoder, predictor = task
                            filename = f"qr_{encoder}_{predictor}_{dataset}_config.yaml"
                        else:
                            predictor = task
                            filename = f"spci_{predictor}_{dataset}_config.yaml"
                        marker = f"{method}-{dataset}-{task_id}"
                        fixture = OmegaConf.create({
                            "model": {"prediction_step": 1, "route_marker": marker},
                            "data": {"data_path": "unused", "route_marker": marker},
                        })
                        path = config_dir / filename
                        OmegaConf.save(config=fixture, f=path)
                        routes.append((job, dataset, task_id, predictor, path, marker))

            for job, dataset, task_id, predictor, expected, marker in routes:
                with self.subTest(config=expected.name):
                    template, config = self._dry_run(job, dataset, task_id, root=root)
                    self.assertEqual(template, expected)
                    self.assertEqual(config.model.route_marker, marker)
                    self.assertEqual(config.data.route_marker, marker)
                    self._assert_artifact(job, config, dataset, predictor, root)


if __name__ == "__main__":
    unittest.main()
