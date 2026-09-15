import io
import unittest

import torch
from omegaconf import OmegaConf

from dscp.loss import compute_loss_rnn_predictor, compute_loss_transformer_predictor
from dscp.models.rnn_predictor import RNNPredictor
from dscp.models.transformer_predictor import TransformerPredictor
from utils.utils import validate_training_quantiles


class _FixedQuantilePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("training_quantiles", torch.tensor([0.2, 0.5, 0.8]))
        self.predictions = torch.nn.Parameter(
            torch.tensor([[-2.0, 1.0, 4.0], [5.0, 0.0, -1.0]])
        )

    def forward(self, x, **kwargs):
        # The loss must supervise the final context, not earlier sequence positions.
        return torch.stack([self.predictions + 100.0, self.predictions], dim=1)


class LocalCPQuantileTrainingTests(unittest.TestCase):
    architectures = ("transformer", "rnn", "gru", "lstm")

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)
        self.history = torch.randn(2, 4, 2)

    @staticmethod
    def _model(architecture, **kwargs):
        options = dict(
            dim_feature=2,
            dim_model=8,
            num_layer=1,
            prediction_step=1,
            dropout=0.0,
            training_quantiles=[0.8, 0.2, 0.5],
        )
        options.update(kwargs)
        if architecture == "transformer":
            return TransformerPredictor(num_head=2, dim_ff=16, **options)
        return RNNPredictor(rnn_type=architecture, **options)

    @staticmethod
    def _loss_fn(architecture):
        return (
            compute_loss_transformer_predictor
            if architecture == "transformer"
            else compute_loss_rnn_predictor
        )

    def _forward(self, architecture, model, current):
        kwargs = dict(current_feature=current, return_repr=True)
        if architecture == "transformer":
            kwargs.update(
                src_mask=torch.nn.Transformer.generate_square_subsequent_mask(
                    self.history.shape[1]
                ),
                src_key_padding_mask=None,
            )
        return model(self.history, **kwargs)

    def test_hand_computed_pinball_loss_and_asymmetric_gradients(self):
        # Errors [[4, 1, -2], [-7, -2, -1]] at taus [.2, .5, .8]
        # produce losses [[.8, .5, .4], [5.6, 1, .2]].
        expected_loss = torch.tensor(8.5 / 6.0)
        expected_gradient = torch.tensor(
            [[-0.2, -0.5, 0.2], [0.8, 0.5, 0.2]]
        ) / 6.0
        for loss_fn in (compute_loss_rnn_predictor, compute_loss_transformer_predictor):
            with self.subTest(loss=loss_fn.__name__):
                model = _FixedQuantilePredictor()
                loss = loss_fn(model, self.history[:, :2, :], torch.tensor([[2.0], [-2.0]]))
                torch.testing.assert_close(loss, expected_loss)
                loss.backward()
                torch.testing.assert_close(model.predictions.grad, expected_gradient)

    def test_quantile_heads_train_the_encoder_with_and_without_current_features(self):
        for architecture in self.architectures:
            for current_dim in (0, 2):
                for levels in ([0.5], [0.8, 0.2, 0.5]):
                    with self.subTest(
                        architecture=architecture, current_dim=current_dim, levels=levels
                    ):
                        model = self._model(
                            architecture,
                            current_feature_dim=current_dim,
                            training_quantiles=levels,
                        )
                        current = torch.randn(2, 1, current_dim) if current_dim else None
                        output, representation = self._forward(architecture, model, current)
                        self.assertEqual(output.shape, (2, 4, len(levels)))
                        self.assertEqual(representation.shape, (2, 4, 8 + current_dim))
                        torch.testing.assert_close(
                            representation, model.encode(model, self.history, current)
                        )
                        before = model.input_linear.weight.detach().clone()
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                        loss = self._loss_fn(architecture)(
                            model, self.history, torch.full((2, 1), 10.0), current
                        )
                        loss.backward()
                        for name, parameter in model.named_parameters():
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        encoder = model.encoder if architecture == "transformer" else model.rnn
                        self.assertGreater(
                            sum(parameter.grad.abs().sum().item() for parameter in encoder.parameters()),
                            0.0,
                        )
                        optimizer.step()
                        self.assertFalse(torch.equal(before, model.input_linear.weight))

                        # Calibration representations do not depend on the training head.
                        encoded_before = model.encode(model, self.history, current).detach().clone()
                        with torch.no_grad():
                            model.output_linear.weight.add_(100.0)
                            model.output_linear.bias.add_(100.0)
                        torch.testing.assert_close(
                            model.encode(model, self.history, current), encoded_before
                        )

    def test_training_levels_are_sorted_and_saved_in_checkpoint(self):
        configured_levels = OmegaConf.create([0.8, 0.2, 0.5])
        self.assertEqual(validate_training_quantiles(configured_levels), [0.2, 0.5, 0.8])
        for architecture in self.architectures:
            with self.subTest(architecture=architecture):
                model = self._model(architecture, training_quantiles=configured_levels)
                self.assertIn("training_quantiles", dict(model.named_buffers()))
                self.assertNotIn("training_quantiles", dict(model.named_parameters()))
                self.assertEqual(model.training_quantiles.dtype, torch.float32)
                torch.testing.assert_close(model.training_quantiles, torch.tensor([0.2, 0.5, 0.8]))
                self.assertEqual(model.output_linear.out_features, 3)
                checkpoint = io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)
                restored = self._model(architecture, training_quantiles=[0.1, 0.4, 0.9])
                restored.load_state_dict(torch.load(checkpoint, weights_only=True))
                torch.testing.assert_close(restored.training_quantiles, model.training_quantiles)
                torch.testing.assert_close(
                    self._forward(architecture, restored, None)[0],
                    self._forward(architecture, model, None)[0],
                )

    def test_invalid_training_levels_are_rejected(self):
        invalid_levels = (
            None, [], 0.5, "0.5", [[0.1, 0.9]], ["0.5"], [True],
            [0.0], [1.0], [-0.1], [1.1], [float("nan")], [float("inf")],
            [0.2, 0.2], [1e-50], [0.999999999], [0.5, 0.5000000001],
        )
        for architecture in ("rnn", "transformer"):
            for levels in invalid_levels:
                with self.subTest(architecture=architecture, levels=levels):
                    with self.assertRaisesRegex(ValueError, "training_quantiles"):
                        self._model(architecture, training_quantiles=levels)

    def test_multi_step_targets_are_rejected(self):
        for architecture in ("rnn", "transformer"):
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(ValueError, "prediction_step=1"):
                    self._model(architecture, prediction_step=2)
                model = self._model(architecture)
                for target in (torch.zeros(2), torch.zeros(2, 3), torch.zeros(2, 1, 1)):
                    with self.assertRaisesRegex(ValueError, "target shape"):
                        self._loss_fn(architecture)(model, self.history, target)


if __name__ == "__main__":
    unittest.main()
