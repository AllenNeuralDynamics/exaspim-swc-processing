"""Tests for :mod:`exaspim_swc_processing.quality_control`."""

import json
import logging
from pathlib import Path

import pytest
from aind_data_schema.core.quality_control import QualityControl

from exaspim_swc_processing.naming import parse_stem
from exaspim_swc_processing.quality_control import (
    cell_from_path,
    discover_quality_control,
    split_by_cell,
)

RESOURCES = Path(__file__).parent / "resources"
STEM = "N001-794492-HP"
OTHER_STEM = "N003-794492-JG"


def _payload() -> dict:
    """Load the fixture record, trimmed from a real QC run.

    Returns
    -------
    dict
        The record as JSON, every metric tagged ``swc: N001-794492-HP``.
    """
    return json.loads((RESOURCES / "quality_control.json").read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> Path:
    """Write a record to ``path``, creating parent directories.

    Parameters
    ----------
    path : Path
        Where to write.
    payload : dict
        The record as JSON.

    Returns
    -------
    Path
        The file written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _untagged(payload: dict) -> dict:
    """Remove the ``swc`` tag from every metric.

    Parameters
    ----------
    payload : dict
        The record as JSON.

    Returns
    -------
    dict
        The record with no metric naming its reconstruction.
    """
    for metric in payload["metrics"]:
        del metric["tags"]["swc"]
    return payload


def _two_cells(payload: dict) -> dict:
    """Reassign the second half of the metrics to another cell, and fail one of them.

    Parameters
    ----------
    payload : dict
        The record as JSON.

    Returns
    -------
    dict
        A record covering two reconstructions with different outcomes.
    """
    half = len(payload["metrics"]) // 2
    for metric in payload["metrics"][half:]:
        metric["tags"]["swc"] = OTHER_STEM
    payload["metrics"][-1]["status_history"][-1]["status"] = "Fail"
    return payload


def test_a_record_is_matched_by_its_swc_tag(tmp_path: Path) -> None:
    """The tag wins over a location that names no reconstruction."""
    _write(tmp_path / "quality_control.json", _payload())
    found = discover_quality_control(tmp_path)
    assert list(found) == [parse_stem(STEM)]
    assert len(found[parse_stem(STEM)].metrics) == len(_payload()["metrics"])


def test_records_are_found_in_nested_directories(tmp_path: Path) -> None:
    """A QC stage may write its records anywhere under the mounted inputs."""
    _write(tmp_path / "qc" / STEM / "quality_control.json", _payload())
    assert list(discover_quality_control(tmp_path)) == [parse_stem(STEM)]


@pytest.mark.parametrize(
    "relative",
    [f"{STEM}_quality_control.json", f"{STEM}/quality_control.json"],
)
def test_untagged_metrics_fall_back_to_the_location(tmp_path: Path, relative: str) -> None:
    """Without the tag, the file or directory name identifies the reconstruction."""
    _write(tmp_path / relative, _untagged(_payload()))
    assert list(discover_quality_control(tmp_path)) == [parse_stem(STEM)]


def test_location_that_names_no_reconstruction_is_none() -> None:
    """Neither the file nor its directory is a reconstruction stem."""
    assert cell_from_path(Path("qc/quality_control.json")) is None


def test_unattributable_metrics_are_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A metric is never published on a cell it may not belong to."""
    payload = _payload()
    payload["metrics"][0]["tags"]["swc"] = "not-a-stem"
    _write(tmp_path / "quality_control.json", payload)
    with caplog.at_level(logging.WARNING):
        found = discover_quality_control(tmp_path)
    assert len(found[parse_stem(STEM)].metrics) == len(payload["metrics"]) - 1
    assert "Dropping quality control metric" in caplog.text


def test_a_file_with_no_attributable_metrics_yields_nothing(tmp_path: Path) -> None:
    """Untagged metrics in a file whose location names no reconstruction."""
    _write(tmp_path / "quality_control.json", _untagged(_payload()))
    assert discover_quality_control(tmp_path) == {}


def test_a_single_cell_record_is_kept_as_is() -> None:
    """Nothing to split, so the record read is the record published."""
    record = QualityControl.model_validate(_payload())
    assert split_by_cell(record, Path("quality_control.json"))[parse_stem(STEM)] is record


def test_a_record_covering_several_cells_is_split(tmp_path: Path) -> None:
    """Each cell gets only its own metrics and a status computed from them."""
    payload = _two_cells(_payload())
    _write(tmp_path / "quality_control.json", payload)
    found = discover_quality_control(tmp_path)
    assert list(found) == [parse_stem(STEM), parse_stem(OTHER_STEM)]

    first, second = found[parse_stem(STEM)], found[parse_stem(OTHER_STEM)]
    assert {metric.tags["swc"] for metric in first.metrics} == {STEM}
    assert {metric.tags["swc"] for metric in second.metrics} == {OTHER_STEM}
    assert len(first.metrics) + len(second.metrics) == len(payload["metrics"])
    assert first.status[f"swc:{STEM}"] == "Pass"
    assert f"swc:{OTHER_STEM}" not in first.status
    assert second.status[f"swc:{OTHER_STEM}"] == "Fail"
    assert first.notes == second.notes == payload["notes"]


def test_records_for_one_cell_in_several_files_are_combined(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two files for the same cell are merged rather than one silently winning."""
    _write(tmp_path / "a" / "quality_control.json", _payload())
    _write(tmp_path / "b" / "quality_control.json", _payload())
    with caplog.at_level(logging.WARNING):
        found = discover_quality_control(tmp_path)
    assert len(found[parse_stem(STEM)].metrics) == 2 * len(_payload()["metrics"])
    assert "Combining quality control" in caplog.text


@pytest.mark.parametrize("content", ["not json", json.dumps({"metrics": "nope"})])
def test_an_invalid_file_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str
) -> None:
    """One bad file does not fail the run."""
    (tmp_path / "broken_quality_control.json").write_text(content, encoding="utf-8")
    _write(tmp_path / "quality_control.json", _payload())
    with caplog.at_level(logging.WARNING):
        found = discover_quality_control(tmp_path)
    assert list(found) == [parse_stem(STEM)]
    assert "not a valid quality control record" in caplog.text


def test_a_directory_matching_the_pattern_is_ignored(tmp_path: Path) -> None:
    """Only files are read."""
    (tmp_path / "quality_control.json").mkdir()
    assert discover_quality_control(tmp_path) == {}


def test_no_records_yields_nothing(tmp_path: Path) -> None:
    """Quality control is optional."""
    assert discover_quality_control(tmp_path) == {}
