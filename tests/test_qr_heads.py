import unittest

import torch

from dscp.loss import (
    compute_loss_quantile_regression_rnn,
    compute_loss_quantile_regression_transformer,
)
from dscp.models.qr_rnn import QuantileRegressionRNN
from dscp.models.qr_transformer import QuantileRegressionTransformer


class QuantileHeadTests(unittest.TestCase):
    architectures = ("transformer", "rnn", "gru", "lstm")
    # Reversed pairs and a repeated endpoint exercise the common level ordering.
    quantile_pairs = [[0.95, 0.05], [0.1, 0.9], [0.05, 0.9]]

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.history = torch.randn(3, 5, 2)

    def _model(self, architecture, **kwargs):
        common = dict(
            dim_feature=2,
            dim_model=8,
            num_layers=1,
            target_quantiles=self.quantile_pairs,
            prediction_step=1,
            dropout=0.0,
            **kwargs,
        )
        if architecture == "transformer":
            return QuantileRegressionTransformer(num_head=2, dim_ff=16, **common)
        return QuantileRegressionRNN(rnn_type=architecture, **common)

    def _predict(self, model, current_feature=None):
        return model.get_predicted_quantile_values(
            model, self.history, current_feature
        )

    def test_default_preserves_nondecreasing_heads_and_checkpoint_keys(self):
        for architecture in self.architectures:
            with self.subTest(architecture=architecture):
                torch.manual_seed(11)
                default = self._model(architecture)
                torch.manual_seed(11)
                explicit = self._model(architecture, head_type="nondecreasing")
                self.assertEqual(default.head_type, "nondecreasing")
                state = default.state_dict()
                self.assertIn("base_head.weight", state)
                self.assertIn("increment_head.weight", state)
                self.assertFalse(any(k.startswith("quantile_heads.") for k in state))
                self.assertEqual(state.keys(), explicit.state_dict().keys())
                for key, value in state.items():
                    torch.testing.assert_close(value, explicit.state_dict()[key])
                explicit.load_state_dict(state, strict=True)
                predicted = self._predict(default)
                torch.testing.assert_close(predicted, self._predict(explicit))
                self.assertTrue(torch.all(predicted[:, 1:] >= predicted[:, :-1]))

    def test_independent_heads_allow_crossing_without_coupling_quantiles(self):
        expected = torch.tensor([3.0, -1.0, 2.0, -2.0]).expand(3, 4)
        for architecture in self.architectures:
            for current_dim in (0, 2):
                with self.subTest(architecture=architecture, current_dim=current_dim):
                    model = self._model(
                        architecture,
                        head_type="independent",
                        current_feature_dim=current_dim,
                    )
                    current = torch.randn(3, 1, current_dim) if current_dim else None
                    self.assertEqual(model.sorted_quantiles, [0.05, 0.1, 0.9, 0.95])
                    self.assertEqual(len(model.quantile_heads), 4)
                    with torch.no_grad():
                        for head, value in zip(model.quantile_heads, expected[0]):
                            head.weight.zero_()
                            head.bias.fill_(value.item())
                    predicted = self._predict(model, current)
                    torch.testing.assert_close(predicted, expected)
                    # Changing a middle head must not alter any other quantile.
                    with torch.no_grad():
                        model.quantile_heads[1].bias.add_(5.0)
                    changed = self._predict(model, current)
                    expected_changed = expected.clone()
                    expected_changed[:, 1] += 5.0
                    torch.testing.assert_close(changed, expected_changed)
                    changed[:, 1].sum().backward()
                    for index, head in enumerate(model.quantile_heads):
                        expected_gradient = 3.0 if index == 1 else 0.0
                        torch.testing.assert_close(
                            head.bias.grad, torch.tensor([expected_gradient])
                        )

    def test_both_modes_train_shared_encoder_and_reload(self):
        targets = torch.full((3, 1), 100.0)
        taus = torch.tensor([0.05, 0.1, 0.9, 0.95])
        for architecture in self.architectures:
            for head_type in ("nondecreasing", "independent"):
                for current_dim in (0, 2):
                    with self.subTest(
                        architecture=architecture,
                        head_type=head_type,
                        current_dim=current_dim,
                    ):
                        model = self._model(
                            architecture,
                            head_type=head_type,
                            current_feature_dim=current_dim,
                        )
                        current = torch.randn(3, 1, current_dim) if current_dim else None
                        predicted = self._predict(model, current)
                        self.assertEqual(predicted.shape, (3, 4))
                        error = targets - predicted
                        expected_loss = torch.where(
                            error >= 0, taus * error, (taus - 1) * error
                        ).mean()
                        loss_fn = (
                            compute_loss_quantile_regression_transformer
                            if architecture == "transformer"
                            else compute_loss_quantile_regression_rnn
                        )
                        loss = loss_fn(
                            model, self.history, targets, self.quantile_pairs, current
                        )
                        torch.testing.assert_close(loss, expected_loss)
                        before = model.input_linear.weight.detach().clone()
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                        loss.backward()
                        for name, parameter in model.named_parameters():
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        optimizer.step()
                        self.assertFalse(torch.equal(before, model.input_linear.weight))
                        restored = self._model(
                            architecture,
                            head_type=head_type,
                            current_feature_dim=current_dim,
                        )
                        restored.load_state_dict(model.state_dict(), strict=True)
                        torch.testing.assert_close(
                            self._predict(restored, current), self._predict(model, current)
                        )

    def test_unknown_head_type_is_rejected(self):
        for architecture in self.architectures:
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(ValueError, "head_type"):
                    self._model(architecture, head_type="typo")


if __name__ == "__main__":
    unittest.main()
