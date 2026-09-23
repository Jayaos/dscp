"""Check that excluded predictions survive interruption and merge without duplication."""

import csv
import json

import numpy as np
import pytest

from baselines.distmatch.exclusions import EXCLUSION_FIELDS, ExclusionJournal, write_excluded_points


def _event(offset=3):
    return {
        "sequence_key": "station,\"one\"", "test_offset": np.int64(offset),
        "target_index": np.int64(40 + offset), "target_y": np.float64(1.5),
        "triggering_quantile_pair": (0.1, 0.9),
        "excluded_quantile_pairs": [[0.1, 0.9], [0.025, 0.975]],
        "lower_bound": np.float32(1.0000001), "upper_bound": np.float32(1),
        "reason": "crossed_bounds", "action": "excluded_from_metrics",
    }


def test_journal_is_readable_immediately_and_survives_later_failure(tmp_path):
    journal = ExclusionJournal({"saving_dir": str(tmp_path), "seed": 7}, "station", "test")
    with pytest.raises(RuntimeError, match="later failure"):
        with journal:
            journal.record(_event())
            first = json.loads(journal.path.read_text(encoding="utf-8"))
            assert first["test_offset"] == 3
            assert first["triggering_quantile_pair"] == [0.1, 0.9]
            journal.record(_event(4))
            raise RuntimeError("later failure")
    events = [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]
    assert [event["test_offset"] for event in events] == [3, 4]


def test_journal_creates_no_files_without_exclusions_or_saving_dir(tmp_path):
    output = tmp_path / "unused"
    with ExclusionJournal({"saving_dir": str(output)}, "station", "test"):
        pass
    assert not output.exists()
    with ExclusionJournal({}, "station", "validation") as journal:
        journal.record(_event())
    assert journal.path is None


def test_concurrent_attempts_for_same_key_have_distinct_journals(tmp_path):
    config = {"saving_dir": str(tmp_path), "seed": 7}
    with ExclusionJournal(config, "station", "test") as first:
        with ExclusionJournal(config, "station", "test") as second:
            first.record(_event(2))
            second.record(_event(5))
            assert first.path != second.path
    assert len(list((tmp_path / "exclusions").glob("*.jsonl"))) == 2
    assert json.loads(first.path.read_text(encoding="utf-8"))["test_offset"] == 2
    assert json.loads(second.path.read_text(encoding="utf-8"))["test_offset"] == 5


def test_csv_uses_completed_log_and_preserves_nested_and_quoted_fields(tmp_path):
    with ExclusionJournal({"saving_dir": str(tmp_path)}, "station", "test") as journal:
        journal.record(_event(99))  # An interrupted attempt is not a result.
    events = [_event(3), _event(4)]
    log = {
        "station": {"metadata": {"excluded_points": events}},
        17: {"metadata": {"excluded_points": [{"test_offset": 8}]}},
        "successful": {"metadata": {"excluded_points": []}},
    }
    path = write_excluded_points(log, tmp_path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert tuple(reader.fieldnames) == EXCLUSION_FIELDS
    assert [row["test_offset"] for row in rows] == ["3", "4", "8"]
    assert rows[0]["sequence_key"] == "station,\"one\""
    assert rows[-1]["sequence_key"] == "17"
    assert json.loads(rows[0]["excluded_quantile_pairs"]) == [[0.1, 0.9], [0.025, 0.975]]
    assert events[0]["triggering_quantile_pair"] == (0.1, 0.9)


def test_empty_csv_has_header_and_failed_replacement_preserves_existing_file(tmp_path):
    destination = write_excluded_points({}, tmp_path)
    original = destination.read_bytes()
    with destination.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert list(reader) == []
        assert tuple(reader.fieldnames) == EXCLUSION_FIELDS
    invalid = {"station": {"metadata": {"excluded_points": [{"lower_bound": np.nan}]}}}
    with pytest.raises(ValueError, match="Out of range float"):
        write_excluded_points(invalid, tmp_path)
    assert destination.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []
