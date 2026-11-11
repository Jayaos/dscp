"""Causal context and independent median-encoder checks."""

import copy
import unittest

import numpy as np
import torch

from conditional_coverage.encoder import (
    _ContextDataset,
    extract_embeddings,
    resolve_model_config,
    resolve_training_config,
    train_encoder,
)


def _prepared():
    index = np.arange(32, dtype=np.float32)
    y = np.sin(index / 4).astype(np.float32)
    return {
        "features": np.column_stack((np.cos(index / 5), y)).astype(np.float32),
        "y_normalized": y,
        "fit_indices": np.arange(3, 17),
        "validation_indices": np.arange(17, 23),
        "calibration_target_indices": np.arange(3, 23),
        "test_indices": np.arange(24, 32),
        "scaler": {},
    }


MODEL = {"rnn_type": "gru", "dim_model": 4, "num_layers": 1, "window_size": 3}
TRAINING = {"batch_size": 8, "learning_rate": 0.01, "epochs": 3, "early_stop": 2}


class ConditionalCoverageEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_median_head_frozen_best_checkpoint_and_unit_embeddings(self):
        prepared = _prepared()
        model, metadata = train_encoder(prepared, MODEL, TRAINING, seed=17)
        self.assertEqual(model.output_linear.out_features, 1)
        self.assertEqual(model.training_quantiles.tolist(), [0.5])
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(metadata["target"], "observed_response")
        self.assertEqual(metadata["best_validation_loss"], min(metadata["validation_loss"]))
        self.assertEqual(metadata["best_epoch"], np.argmin(metadata["validation_loss"]) + 1)
        self.assertEqual(metadata["fit_count"], len(prepared["fit_indices"]))
        valid = _ContextDataset(prepared["features"], prepared["validation_indices"], 3,
                                prepared["y_normalized"])
        contexts = torch.stack([valid[index][0] for index in range(len(valid))])
        targets = torch.stack([valid[index][1] for index in range(len(valid))])
        restored_loss = float(0.5 * torch.abs(targets - model(contexts)[:, -1, 0]).mean())
        self.assertAlmostEqual(restored_loss, metadata["best_validation_loss"], places=6)
        embeddings = extract_embeddings(model, prepared, prepared["test_indices"], batch_size=3)
        self.assertEqual(embeddings.shape, (8, 4))
        self.assertEqual(embeddings.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)

    def test_default_lstm_architecture_can_fit_and_encode_observed_targets(self):
        model, _ = train_encoder(_prepared(), {"dim_model": 4, "window_size": 3},
                                 {**TRAINING, "epochs": 1}, seed=4)
        self.assertIsInstance(model.rnn, torch.nn.LSTM)
        result = extract_embeddings(model, _prepared(), np.array([24, 25]))
        self.assertEqual(result.shape, (2, 4))

    def test_seed_is_reproducible_and_training_restores_global_torch_rng(self):
        state = torch.random.get_rng_state().clone()
        first, first_metadata = train_encoder(_prepared(), MODEL, TRAINING, seed=19)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        second, second_metadata = train_encoder(_prepared(), MODEL, TRAINING, seed=19)
        self.assertEqual(first_metadata, second_metadata)
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

    def test_test_observations_do_not_affect_fitting_or_checkpoint_selection(self):
        prepared = _prepared()
        changed = copy.deepcopy(prepared)
        changed["features"][23:] = 100000
        changed["y_normalized"][23:] = np.nan
        first, first_metadata = train_encoder(prepared, MODEL, TRAINING, seed=2)
        second, second_metadata = train_encoder(changed, MODEL, TRAINING, seed=2)
        self.assertEqual(first_metadata, second_metadata)
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

    def test_embedding_reads_only_context_and_does_not_require_target_array(self):
        prepared = _prepared()
        model, _ = train_encoder(prepared, MODEL, TRAINING)
        first = extract_embeddings(model, prepared, np.array([24]))
        changed = {"features": prepared["features"].copy()}
        changed["features"][24:] = np.nan
        same = extract_embeddings(model, changed, np.array([24]))
        np.testing.assert_array_equal(first, same)
        changed["features"][23] += 10
        different = extract_embeddings(model, changed, np.array([24]))
        self.assertFalse(np.allclose(first, different))

    def test_dataset_uses_observed_response_at_target_and_only_preceding_history(self):
        prepared = _prepared()
        dataset = _ContextDataset(prepared["features"], np.array([8]), 3, prepared["y_normalized"])
        context, target = dataset[0]
        np.testing.assert_array_equal(context.numpy(), prepared["features"][5:8])
        self.assertEqual(float(target), float(prepared["y_normalized"][8]))

    def test_collapsed_embeddings_raise_instead_of_silently_normalizing(self):
        model, _ = train_encoder(_prepared(), MODEL, TRAINING)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        with self.assertRaisesRegex(ValueError, "zero or near-zero"):
            extract_embeddings(model, _prepared(), np.array([24]))

    def test_crossed_fit_validation_or_test_overlap_are_rejected(self):
        prepared = _prepared()
        prepared["validation_indices"] = np.arange(16, 23)
        with self.assertRaisesRegex(ValueError, "chronologically"):
            train_encoder(prepared, MODEL, TRAINING)
        prepared = _prepared()
        prepared["test_indices"] = np.arange(22, 32)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            train_encoder(prepared, MODEL, TRAINING)

    def test_configuration_validation(self):
        for config in ({"window_size": True}, {"dim_model": 0}, {"dropout": 1},
                       {"rnn_type": "bidirectional"}, {"target_quantiles": [0.5]}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_model_config(config)
        for config in ({"epochs": 0}, {"learning_rate": float("nan")}, {"batch_size": 1.5},
                       {"early_stop": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_training_config(config)


if __name__ == "__main__":
    unittest.main()
