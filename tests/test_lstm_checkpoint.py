import contextlib
import io
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from base_predictor import lstm_predictor
from sbatch_run_base_predictor import run_lstm


class LSTMCheckpointTests(unittest.TestCase):
    def test_checkpoint_reproduces_saved_forecasts_and_supports_latent_gradients(self):
        # Unequal series and a constant covariate exercise pooled scaling and epsilon.
        data = SimpleNamespace(data_type="toy", data={})
        for key, length, offset in (("a", 24, 0.0), ("b", 30, 3.0)):
            t = np.arange(length, dtype=np.float32)
            data.data[key] = {
                "x": np.column_stack((t + offset, np.ones_like(t))),
                "y": np.sin(t / 4) + offset,
            }
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(lstm_predictor, "tqdm", side_effect=lambda values: values):
            torch.manual_seed(17)
            predictor = lstm_predictor.LSTMPredictor(
                data, embedding_dim=3, hidden_dim=4, num_layers=2,
                train_ratio=0.6, window_length=3,
            )
            predictor.fit_predict(
                train_ratio=0.75, batch_size=8, learning_rate=0.001,
                max_epoch=2, early_stop=1, device="cpu",
            )
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                predictor.save(directory)
                model_path = directory / "lstm_toy_model.pt"
                self.assertFalse(model_path.exists())
                with (directory / "lstm_toy_data.pkl").open("rb") as stream:
                    original_artifact = pickle.load(stream)

                predictor.save(directory, save_model=True)
                model, checkpoint = lstm_predictor.load_lstm_checkpoint(model_path)
                self.assertFalse(model.training)
                self.assertEqual(checkpoint["metadata"]["best_epoch"], predictor.best_epoch)
                self.assertEqual(checkpoint["metadata"]["window_length"], 3)
                self.assertEqual(checkpoint["metadata"]["input_order"], ["covariates", "lagged_target"])
                for name, value in predictor.lstm.state_dict().items():
                    self.assertEqual(checkpoint["model_state_dict"][name].device.type, "cpu")
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

                stats = checkpoint["normalization"]
                epsilon = stats["epsilon"]
                self.assertEqual(stats["x_std"][1].item(), 0.0)
                with (directory / "lstm_toy_data.pkl").open("rb") as stream:
                    artifact = pickle.load(stream)
                for key, item in artifact.items():
                    with self.subTest(series=key):
                        for field in item:
                            np.testing.assert_array_equal(item[field], original_artifact[key][field])
                        normalized = {
                            field: (item[field] - stats[f"{kind}_mean"].numpy())
                            / (stats[f"{kind}_std"].numpy() + epsilon)
                            for field, kind in (
                                ("train_x", "x"), ("heldout_x", "x"),
                                ("train_y", "y"), ("heldout_y", "y"),
                            )
                        }
                        windows, _ = lstm_predictor._make_heldout_sequence_prediction_data(
                            **normalized, k=checkpoint["metadata"]["window_length"],
                        )
                        inputs = torch.from_numpy(windows).float()
                        with torch.no_grad():
                            restored_predictions = model(inputs)[:, -1, :]
                            restored_predictions = (
                                restored_predictions * (stats["y_std"] + epsilon) + stats["y_mean"]
                            )
                        np.testing.assert_allclose(
                            restored_predictions.numpy(), item["heldout_predictions"],
                            rtol=1e-5, atol=1e-6,
                        )

                model.requires_grad_(False)
                features = model.encode(inputs)[:, -1, :].detach().requires_grad_(True)
                model.output_fc(features).sum().backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertGreater(features.grad.abs().sum().item(), 0)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_cli_forwards_optional_save_flag(self):
        for enabled in (False, True):
            with self.subTest(save_model=enabled), \
                    patch.object(run_lstm, "BasePredictorData"), \
                    patch.object(run_lstm, "LSTMPredictor") as predictor_class, \
                    patch("sys.argv", ["run_lstm", "toy", "--save-dir", "output"]
                          + (["--save-model"] if enabled else [])), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_lstm.main()
                predictor_class.return_value.save.assert_called_once_with(
                    "output", save_model=enabled,
                )


if __name__ == "__main__":
    unittest.main()
