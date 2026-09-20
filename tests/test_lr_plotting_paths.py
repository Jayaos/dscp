import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from sbatch.sbatch_run_plotting import run_lr_plotting


class LRPlottingPathTests(unittest.TestCase):
    def test_artifact_mapping_matches_base_predictor_outputs(self):
        self.assertEqual(
            run_lr_plotting.LR_ARTIFACTS,
            {
                "air-10": Path(
                    "data/air-10_prediction/lr/lr_air-10_data.pkl"
                ),
                "nsdb-60m": Path(
                    "data/solar_prediction/lr/lr_nsdb-60m_data.pkl"
                ),
                "sapflux-solo3-large": Path(
                    "data/sapflux-solo3-large/lr/"
                    "lr_sapflux-solo3-large_data.pkl"
                ),
            },
        )

    def test_sapflux_cli_uses_correct_artifact_without_io_or_plotting(self):
        dataset = "sapflux-solo3-large"
        repository = Path("C:/unused_lr_plotting_repository")
        artifact_path = (
            repository
            / "data"
            / "sapflux-solo3-large"
            / "lr"
            / "lr_sapflux-solo3-large_data.pkl"
        )
        prediction_data = {
            "series": {
                "heldout_y": [1.0, 2.0],
                "heldout_predictions": [0.9, 2.1],
            }
        }

        with (
            patch.object(run_lr_plotting, "REPO_ROOT", repository),
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                run_lr_plotting,
                "load_data",
                return_value=prediction_data,
            ) as load_data,
            patch.object(
                run_lr_plotting,
                "plot_darts_predictions",
            ) as plot_predictions,
            patch.object(
                sys,
                "argv",
                [
                    "run_lr_plotting.py",
                    "--datasets",
                    dataset,
                    "--plot-len",
                    "2",
                    "--n-seqs",
                    "1",
                    "--output-root",
                    "plots",
                ],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_lr_plotting.main()

        load_data.assert_called_once_with(str(artifact_path))
        plot_predictions.assert_called_once_with(
            prediction_data,
            plot_len=2,
            n_seqs=1,
            save_dir=str(repository / "plots" / dataset),
        )


if __name__ == "__main__":
    unittest.main()
