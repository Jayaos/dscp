"""Verify local shard execution and refusal to publish incomplete/mixed runs."""

from contextlib import contextmanager
import copy
import csv
import os
from pathlib import Path
import pickle

import numpy as np
from omegaconf import OmegaConf
import pytest

from baselines.distmatch.distributed import merge_shards, prepare_shards, run_shard
from baselines.distmatch.model import DistMatchCrossedBoundsError, DistMatchResidualIntervalEstimator
from baselines.distmatch.run_distmatch import run_distmatch


def _write_pickle(path, value):
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def _experiment(directory, count=3, plotting=False):
    keys = ["station_z", 17, "station_a"] if count == 3 else [f"station_{index}" for index in range(count)]
    data = {}
    for index, key in enumerate(keys):
        t = np.arange(18 + index % 3, dtype=float)
        prediction = 20 + index + 0.2 * t
        data[key] = {
            "heldout_y": prediction + 1.2 * np.sin(0.7 * t + index),
            "heldout_predictions": prediction[:, None],
        }
    artifact_path = directory / "forecast.pkl"
    _write_pickle(artifact_path, data)
    config = {
        "seed": 37, "num_cores": 1, "threads_per_worker": 1, "show_progress": False,
        "data": {
            "data_path": str(artifact_path), "train_ratio": 0.5,
            "valid_ratio": 0.2, "test_ratio": 0.3, "normalize_residual": True,
        },
        "model": {
            "past_window_len": 3, "match_threshold": 0.5,
            "n_trees": 2, "qrf_n_estimators": 2, "qrf_max_depth": 2,
            "beta_bins": 3, "target_quantiles": [[0.1, 0.9]],
        },
        "plotting": {"plotting": plotting, "plotting_seq_len": 4},
        "saving_dir": str(directory / "sharded"),
    }
    return data, config


def _assert_not_published(config):
    output = Path(config["saving_dir"])
    for name in ("log.pkl", "summary_results.pkl", "run_metadata.yaml", "excluded_points.csv", "plots"):
        assert not (output / name).exists(), f"Published {name} before validating all shards"


@contextmanager
def _modified_pickle(path, value):
    original = path.read_bytes()
    try:
        _write_pickle(path, value)
        yield
    finally:
        path.write_bytes(original)


@pytest.fixture(scope="module")
def completed_shards(tmp_path_factory):
    directory = tmp_path_factory.mktemp("distmatch_completed_shards")
    data, config = _experiment(directory)
    manifest = prepare_shards(config, 2)
    envelopes = [run_shard(config, index) for index in range(2)]
    return data, config, manifest, envelopes


@pytest.mark.parametrize("count,expected_sizes", [(50, [25, 25]), (5, [3, 2])])
def test_balanced_shards_cover_artifact_order_without_duplicates(tmp_path, count, expected_sizes):
    data, config = _experiment(tmp_path, count)
    manifest = prepare_shards(config, 2)
    assert manifest["num_shards"] == 2
    assert manifest["num_sequences"] == count
    assert manifest["sequence_keys"] == list(data)
    assert [len(keys) for keys in manifest["shard_keys"]] == expected_sizes
    assert [key for keys in manifest["shard_keys"] for key in keys] == list(data)
    assert len(set(key for keys in manifest["shard_keys"] for key in keys)) == count
    assert _read_pickle(Path(config["saving_dir"]) / "manifest.pkl") == manifest
    _assert_not_published(config)


def test_two_local_shards_match_ordinary_run_and_publish_standard_outputs(tmp_path):
    data, config = _experiment(tmp_path, plotting=True)
    ordinary_config = copy.deepcopy(config)
    ordinary_config["saving_dir"] = str(tmp_path / "ordinary")
    config_path = tmp_path / "ordinary.yaml"
    OmegaConf.save(OmegaConf.create(ordinary_config), config_path)
    expected = run_distmatch(config_path)

    manifest = prepare_shards(config, 2)
    # Completion order must not affect final key order or per-sequence seeds.
    for index in (1, 0):
        envelope = run_shard(config, index)
        assert envelope["sequence_keys"] == manifest["shard_keys"][index]
        assert list(envelope["log"]) == manifest["shard_keys"][index]
        assert envelope["run_id"] == manifest["run_id"]
    _assert_not_published(config)
    actual = merge_shards(config)
    assert list(actual) == list(expected) == list(data)
    for key in data:
        assert actual[key]["evaluation_results"] == expected[key]["evaluation_results"]
        for field in ("seed", "sequence_seed", "normalization", "target_indices", "diagnostics"):
            assert actual[key]["metadata"][field] == expected[key]["metadata"][field]

    output = Path(config["saving_dir"])
    for name in ("log.pkl", "summary_results.pkl", "resolved_config.yaml", "run_metadata.yaml", "excluded_points.csv"):
        assert (output / name).is_file()
    assert _read_pickle(output / "log.pkl") == actual
    assert _read_pickle(output / "summary_results.pkl") == _read_pickle(tmp_path / "ordinary" / "summary_results.pkl")
    assert len(list((output / "plots").glob("*.pdf"))) == len(data)
    resolved = OmegaConf.load(output / "resolved_config.yaml")
    assert resolved.saving_dir == str(output)
    assert resolved.num_cores == config["num_cores"]
    assert resolved.data.normalize_residual is True
    metadata = OmegaConf.load(output / "run_metadata.yaml")
    assert metadata.num_sequences == len(data)
    assert metadata.configured_num_cores == 2 * config["num_cores"]
    assert metadata.effective_num_cores == 2
    assert metadata.distributed.run_id == manifest["run_id"]
    assert list(metadata.distributed.sequence_counts) == [2, 1]
    assert list(metadata.distributed.effective_workers_per_shard) == [1, 1]


def test_exclusions_from_all_shards_are_merged_once_in_original_order(tmp_path, monkeypatch):
    data, config = _experiment(tmp_path)
    predict = DistMatchResidualIntervalEstimator.predict_intervals

    def cross_first_prediction(self, pairs):
        if not getattr(self, "_first_prediction_seen", False):
            self._first_prediction_seen = True
            raise DistMatchCrossedBoundsError(
                triggering_quantile_pair=pairs[0], tree_index=0, beta=0.1,
                lower_quantile=0.1, upper_quantile=0.9,
                lower_bound=1.0000001, upper_bound=1.0,
            )
        return predict(self, pairs)

    monkeypatch.setattr(DistMatchResidualIntervalEstimator, "predict_intervals", cross_first_prediction)
    prepare_shards(config, 2)
    run_shard(config, 1)
    run_shard(config, 0)
    output = Path(config["saving_dir"])
    assert len(list((output / "exclusions").glob("*.jsonl"))) == len(data)
    _assert_not_published(config)
    log = merge_shards(config)
    with (output / "excluded_points.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["sequence_key"] for row in rows] == [str(key) for key in data]
    assert [row["test_offset"] for row in rows] == ["0"] * len(data)
    for entry in log.values():
        assert len(entry["metadata"]["excluded_points"]) == 1
        assert entry["metadata"]["valid_prediction_mask"][0] is False
        for result in entry["evaluation_results"].values():
            assert result["target_indices"] == entry["metadata"]["target_indices"][1:]


def test_existing_exclusions_csv_prevents_merge_overwrite(completed_shards):
    _, config, _, _ = completed_shards
    path = Path(config["saving_dir"]) / "excluded_points.csv"
    sentinel = b"Existing exclusions must remain unchanged\n"
    try:
        path.write_bytes(sentinel)
        with pytest.raises(FileExistsError, match="already exist"):
            merge_shards(config)
        assert path.read_bytes() == sentinel
        assert not (path.parent / "log.pkl").exists()
    finally:
        path.unlink()


@pytest.mark.parametrize("count", [None, True, -1, 0, 1, 1.5, "2", 4])
def test_invalid_shard_counts_do_not_create_run_directory(tmp_path, count):
    _, config = _experiment(tmp_path)
    with pytest.raises(ValueError):
        prepare_shards(config, count)
    assert not Path(config["saving_dir"]).exists()


@pytest.mark.parametrize("index", [None, True, -1, 2, 0.5, "0"])
def test_invalid_shard_indices_are_rejected(tmp_path, index):
    _, config = _experiment(tmp_path)
    prepare_shards(config, 2)
    with pytest.raises(ValueError):
        run_shard(config, index)
    assert not list((Path(config["saving_dir"]) / "shards").glob("*.pkl"))
    _assert_not_published(config)


def test_preparation_requires_fresh_root_and_preserves_existing_results(tmp_path):
    _, config = _experiment(tmp_path)
    output = Path(config["saving_dir"])
    output.mkdir()
    sentinel = output / "log.pkl"
    sentinel.write_bytes(b"Existing run must remain unchanged")
    with pytest.raises((ValueError, FileExistsError)):
        prepare_shards(config, 2)
    assert sentinel.read_bytes() == b"Existing run must remain unchanged"
    assert not (output / "manifest.pkl").exists()


def test_prepared_run_cannot_be_prepared_again_or_completed_shard_reexecuted(completed_shards):
    _, config, manifest, _ = completed_shards
    output = Path(config["saving_dir"])
    before = (output / "shards" / "shard_0000.pkl").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        prepare_shards(config, 2)
    with pytest.raises((ValueError, FileExistsError)):
        run_shard(config, 0)
    assert _read_pickle(output / "manifest.pkl") == manifest
    assert (output / "shards" / "shard_0000.pkl").read_bytes() == before
    _assert_not_published(config)


def test_failed_shard_evaluation_publishes_nothing_and_releases_claim(tmp_path, monkeypatch):
    _, config = _experiment(tmp_path)
    prepare_shards(config, 2)

    def fail_evaluation(*args, **kwargs):
        raise RuntimeError("simulated worker failure")

    monkeypatch.setattr("baselines.distmatch.distributed.evaluate_sequences", fail_evaluation)
    with pytest.raises(RuntimeError, match="simulated worker failure"):
        run_shard(config, 0)
    assert list((Path(config["saving_dir"]) / "shards").iterdir()) == []
    _assert_not_published(config)


@pytest.mark.parametrize("corruption", [
    "wrong_run", "wrong_config", "wrong_artifact", "wrong_index", "wrong_assignment",
    "duplicate_sequence", "missing_sequence", "wrong_schema",
])
def test_invalid_shard_envelopes_are_rejected_before_combined_publication(completed_shards, corruption):
    _, config, manifest, envelopes = completed_shards
    envelope = copy.deepcopy(envelopes[1])
    if corruption == "wrong_run":
        envelope["run_id"] = "another-run"
    elif corruption == "wrong_config":
        envelope["config"]["seed"] += 1
    elif corruption == "wrong_artifact":
        envelope["artifact"]["size"] += 1
    elif corruption == "wrong_index":
        envelope["shard_index"] = 0
    elif corruption == "wrong_assignment":
        envelope["sequence_keys"] = manifest["shard_keys"][0]
    elif corruption == "duplicate_sequence":
        key = manifest["shard_keys"][0][0]
        envelope["log"][key] = envelopes[0]["log"][key]
    elif corruption == "missing_sequence":
        envelope["log"].pop(manifest["shard_keys"][1][0])
    else:
        envelope["schema_version"] = -1
    path = Path(config["saving_dir"]) / "shards" / "shard_0001.pkl"
    with _modified_pickle(path, envelope):
        with pytest.raises(ValueError):
            merge_shards(config)
        _assert_not_published(config)


def test_missing_shard_is_rejected_before_combined_publication(completed_shards):
    _, config, _, _ = completed_shards
    path = Path(config["saving_dir"]) / "shards" / "shard_0001.pkl"
    original = path.read_bytes()
    try:
        path.unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            merge_shards(config)
        _assert_not_published(config)
    finally:
        path.write_bytes(original)


def test_truncated_shard_is_rejected_before_combined_publication(completed_shards):
    _, config, _, _ = completed_shards
    path = Path(config["saving_dir"]) / "shards" / "shard_0001.pkl"
    original = path.read_bytes()
    try:
        path.write_bytes(b"\x80\x05")
        with pytest.raises(ValueError):
            merge_shards(config)
        _assert_not_published(config)
    finally:
        path.write_bytes(original)


def test_unexpected_shard_is_rejected_before_combined_publication(completed_shards):
    _, config, _, envelopes = completed_shards
    path = Path(config["saving_dir"]) / "shards" / "shard_0002.pkl"
    try:
        _write_pickle(path, envelopes[0])
        with pytest.raises(ValueError):
            merge_shards(config)
        _assert_not_published(config)
    finally:
        path.unlink()


@pytest.mark.parametrize("stage", ["run", "merge"])
def test_changed_run_config_is_rejected_before_publication(completed_shards, stage):
    _, config, _, _ = completed_shards
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    with pytest.raises(ValueError):
        run_shard(changed, 0) if stage == "run" else merge_shards(changed)
    _assert_not_published(config)


@pytest.mark.parametrize("stage", ["run", "merge"])
def test_changed_artifact_is_rejected_before_publication(completed_shards, stage):
    _, config, _, _ = completed_shards
    path = Path(config["data"]["data_path"])
    original, stat = path.read_bytes(), path.stat()
    try:
        path.write_bytes(original + b"changed")
        with pytest.raises(ValueError):
            run_shard(config, 0) if stage == "run" else merge_shards(config)
        _assert_not_published(config)
    finally:
        path.write_bytes(original)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
