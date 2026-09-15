import unittest

import torch

from dscp.loss import compute_loss_iqn_rnn, compute_loss_iqn_transformer
from dscp.models.iqn import (
    ImplicitQuantileNetwork,
    PartiallyMonotonicQuantileHead,
    build_iqn_optimizer,
    build_quantile_head,
)
from dscp.models.iqn_rnn import IQNRNN
from dscp.models.iqn_transformer import IQNTransformer


class IQNPredictionHeadTests(unittest.TestCase):
    architectures = ("transformer", "rnn", "gru", "lstm")

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.history = torch.randn(4, 6, 3)
        self.quantiles = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])

    def _model(self, architecture, **kwargs):
        common = dict(
            dim_feature=3,
            dim_model=8,
            num_layers=1,
            iqn_hidden_dim=7,
            n_cos_embedding=9,
            dropout=0.0,
            **kwargs,
        )
        if architecture == "transformer":
            return IQNTransformer(num_head=2, dim_ff=16, **common)
        return IQNRNN(rnn_type=architecture, **common)

    def _predict(self, model, current_feature=None, sampling_num=64):
        model.eval()
        return model.get_predicted_quantile_values(
            model,
            self.history,
            self.quantiles,
            current_feature=current_feature,
            sampling_num=sampling_num,
        )

    def test_default_preserves_cosine_embedding_head_and_checkpoint_keys(self):
        for architecture in self.architectures:
            with self.subTest(architecture=architecture):
                torch.manual_seed(11)
                default = self._model(architecture)
                torch.manual_seed(11)
                explicit = self._model(
                    architecture,
                    prediction_head="cosine_embedding",
                )

                self.assertEqual(default.prediction_head, "cosine_embedding")
                self.assertIsInstance(default.iqn, ImplicitQuantileNetwork)
                state = default.state_dict()
                self.assertIn(
                    "iqn.quantile_embedding.output_layer.0.weight",
                    state,
                )
                self.assertEqual(state.keys(), explicit.state_dict().keys())
                for key, value in state.items():
                    torch.testing.assert_close(value, explicit.state_dict()[key])

                explicit.load_state_dict(state, strict=True)
                torch.manual_seed(23)
                expected = self._predict(default)
                torch.manual_seed(23)
                torch.testing.assert_close(expected, self._predict(explicit))

    def test_partially_monotonic_head_is_nondecreasing_after_updates(self):
        for hidden_dims in ([9], [9, 7, 5]):
            for activation in ("tanh", "sigmoid", "softplus"):
                with self.subTest(hidden_dims=hidden_dims, activation=activation):
                    head = PartiallyMonotonicQuantileHead(
                        input_dim=4,
                        hidden_dims=hidden_dims,
                        activation=activation,
                    )
                    contexts = torch.randn(6, 4)
                    taus = torch.linspace(0.001, 0.999, 501)
                    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)

                    for _ in range(3):
                        values, _ = head(contexts, taus=taus)
                        self.assertEqual(values.shape, (6, 501))
                        self.assertTrue(
                            torch.all(values[:, 1:] >= values[:, :-1] - 1e-6)
                        )
                        optimizer.zero_grad()
                        values[:, ::50].square().mean().backward()
                        optimizer.step()

                    self.assertTrue(torch.all(head.tau_weight > 0))
                    self.assertTrue(torch.all(head.output_weight > 0))
                    for layer in head.positive_hidden_layers:
                        self.assertTrue(torch.all(layer.weight > 0))

    def test_monotonic_prediction_is_direct_and_does_not_consume_rng(self):
        head = PartiallyMonotonicQuantileHead(
            input_dim=4,
            hidden_dims=[8, 6],
            activation="tanh",
        )
        contexts = torch.randn(3, 4)
        expected, prepared_taus = head(contexts, taus=self.quantiles)

        rng_before = torch.random.get_rng_state().clone()
        actual = head.predict_quantiles(
            contexts,
            self.quantiles,
            sampling_num=1,
        )
        rng_after = torch.random.get_rng_state()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            prepared_taus,
            self.quantiles.unsqueeze(0).expand(3, -1),
        )
        torch.testing.assert_close(rng_before, rng_after)
        self.assertTrue(torch.all(actual[:, 1:] >= actual[:, :-1] - 1e-6))

    def test_monotonic_head_outputs_are_not_constrained_to_be_nonnegative(self):
        head = PartiallyMonotonicQuantileHead(
            input_dim=3,
            hidden_dims=[5, 4],
        )
        with torch.no_grad():
            head.base_layer.weight.zero_()
            head.base_layer.bias.fill_(-10.0)

        values, _ = head(torch.randn(2, 3), taus=self.quantiles)
        self.assertTrue(torch.all(values < 0))
        self.assertTrue(torch.all(values[:, 1:] >= values[:, :-1] - 1e-6))

    def test_wrappers_select_and_directly_evaluate_monotonic_head(self):
        for architecture in self.architectures:
            for current_dim in (0, 2):
                with self.subTest(
                    architecture=architecture,
                    current_dim=current_dim,
                ):
                    model = self._model(
                        architecture,
                        current_feature_dim=current_dim,
                        prediction_head="partially_monotonic",
                        monotonic_num_layers=3,
                        monotonic_activation="tanh",
                    )
                    current = (
                        torch.randn(4, 1, current_dim) if current_dim else None
                    )
                    self.assertEqual(
                        model.prediction_head,
                        "partially_monotonic",
                    )
                    self.assertIsInstance(
                        model.iqn,
                        PartiallyMonotonicQuantileHead,
                    )
                    self.assertEqual(model.iqn.hidden_dims, (7, 7, 7))

                    actual = self._predict(model, current, sampling_num=1)
                    if architecture == "transformer":
                        mask = torch.nn.Transformer.generate_square_subsequent_mask(
                            self.history.shape[1]
                        )
                        expected, _ = model(
                            self.history,
                            current_feature=current,
                            taus=self.quantiles,
                            src_mask=mask,
                            src_key_padding_mask=None,
                        )
                    else:
                        expected, _ = model(
                            self.history,
                            current_feature=current,
                            taus=self.quantiles,
                        )
                    torch.testing.assert_close(actual, expected)
                    self.assertTrue(
                        torch.all(actual[:, 1:] >= actual[:, :-1] - 1e-6)
                    )

    def test_sampled_pinball_training_reaches_encoder_and_monotonic_head(self):
        targets = torch.randn(4, 1)
        for architecture in self.architectures:
            with self.subTest(architecture=architecture):
                model = self._model(
                    architecture,
                    prediction_head="partially_monotonic",
                    monotonic_num_layers=2,
                )
                loss_fn = (
                    compute_loss_iqn_transformer
                    if architecture == "transformer"
                    else compute_loss_iqn_rnn
                )
                loss = loss_fn(model, self.history, targets, num_taus=11)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()

                self.assertIsNotNone(model.input_linear.weight.grad)
                for name, parameter in model.iqn.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_optimizer_does_not_decay_raw_positive_parameters_toward_zero(self):
        monotonic_model = self._model(
            "rnn",
            prediction_head="partially_monotonic",
            monotonic_num_layers=3,
        )
        optimizer = build_iqn_optimizer(
            monotonic_model,
            learning_rate=0.001,
            weight_decay=0.125,
        )
        self.assertEqual(len(optimizer.param_groups), 2)

        zero_decay_group = next(
            group for group in optimizer.param_groups
            if group["weight_decay"] == 0.0
        )
        decayed_group = next(
            group for group in optimizer.param_groups
            if group["weight_decay"] == 0.125
        )
        positive_ids = {
            id(parameter)
            for parameter in monotonic_model.iqn.positive_raw_parameters()
        }
        self.assertEqual(
            {id(parameter) for parameter in zero_decay_group["params"]},
            positive_ids,
        )
        self.assertTrue(
            positive_ids.isdisjoint(
                id(parameter) for parameter in decayed_group["params"]
            )
        )

        cosine_model = self._model("rnn", prediction_head="cosine_embedding")
        cosine_optimizer = build_iqn_optimizer(
            cosine_model,
            learning_rate=0.001,
            weight_decay=0.125,
        )
        self.assertEqual(len(cosine_optimizer.param_groups), 1)
        self.assertEqual(cosine_optimizer.param_groups[0]["weight_decay"], 0.125)

        direct_optimizer = build_iqn_optimizer(
            monotonic_model.iqn,
            learning_rate=0.001,
            weight_decay=0.125,
        )
        direct_zero_decay = next(
            group for group in direct_optimizer.param_groups
            if group["weight_decay"] == 0.0
        )
        self.assertEqual(
            {id(parameter) for parameter in direct_zero_decay["params"]},
            positive_ids,
        )

    def test_level_domain_shapes_and_selector_validation(self):
        head = PartiallyMonotonicQuantileHead(input_dim=3, hidden_dims=[4])
        contexts = torch.randn(2, 3)

        scalar_values, scalar_taus = head(contexts, taus=0.5)
        self.assertEqual(scalar_values.shape, (2, 1))
        self.assertEqual(scalar_taus.shape, (2, 1))

        matrix_taus = torch.tensor([[0.1, 0.8], [0.2, 0.9]])
        matrix_values, returned_taus = head(contexts, taus=matrix_taus)
        self.assertEqual(matrix_values.shape, (2, 2))
        torch.testing.assert_close(returned_taus, matrix_taus)

        sampled = head.sample_taus(32, 17, contexts.device, contexts.dtype)
        self.assertTrue(torch.all((sampled > 0) & (sampled < 1)))

        for invalid in (0.0, 1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    head(contexts, taus=invalid)

        with self.assertRaisesRegex(ValueError, "prediction_head"):
            build_quantile_head("typo", input_dim=3)
        with self.assertRaisesRegex(ValueError, "activation"):
            PartiallyMonotonicQuantileHead(
                input_dim=3,
                hidden_dims=[4],
                activation="gelu",
            )
        with self.assertRaisesRegex(ValueError, "hidden_dims"):
            PartiallyMonotonicQuantileHead(input_dim=3, hidden_dims=[])
        with self.assertRaisesRegex(TypeError, "positive integer"):
            PartiallyMonotonicQuantileHead(input_dim=3, hidden_dims=[4.5])

        explicit_widths = build_quantile_head(
            "partially_monotonic",
            input_dim=3,
            hidden_dim=99,
            monotonic_num_layers=7,
            monotonic_hidden_dims=[6, 4, 2],
        )
        self.assertEqual(explicit_widths.hidden_dims, (6, 4, 2))


if __name__ == "__main__":
    unittest.main()
