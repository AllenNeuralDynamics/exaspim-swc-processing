"""Tests for :mod:`exaspim_swc_processing.quality_control`."""

import json
import logging
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest
from aind_data_schema.core.quality_control import QualityControl

from exaspim_swc_processing.naming import parse_stem
from exaspim_swc_processing.quality_control import (
    cell_from_path,
    discover_quality_control,
    resolve_reference,
    resolve_skeleton_references,
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


CCF_SOURCE = "ccf_space_reconstructions/reconstructions.precomputed"
SPECIMEN_SOURCE = "specimen_space_reconstructions/reconstructions.precomputed"
ASSET_URI = "s3://aind-open-data/exaSPIM_794492_x_reconstruction-N001_y"
IMAGE = "zarr://s3://aind-open-data/exaSPIM_794492 x/fused.zarr"


def _reference(*params: tuple[str, str]) -> str:
    """Encode a reference the way ``standard_morph`` does.

    Parameters
    ----------
    *params : tuple[str, str]
        Query parameters, in order.

    Returns
    -------
    str
        The ``js:neuroglancer?`` reference.
    """
    return "js:neuroglancer?" + urlencode(params, quote_via=quote, safe=":/,")


def test_a_relative_skeleton_is_made_absolute() -> None:
    """Only the skeleton changes; the image and goto actions survive byte for byte."""
    reference = _reference(
        ("image", IMAGE),
        ("skeleton", CCF_SOURCE),
        ("segments", "7"),
        ("action", "goto:1.5,2,3"),
        ("action", "goto:4,5,6"),
    )
    assert resolve_reference(reference, ASSET_URI, {CCF_SOURCE}) == _reference(
        ("image", IMAGE),
        ("skeleton", f"{ASSET_URI}/{CCF_SOURCE}"),
        ("segments", "1"),
        ("action", "goto:1.5,2,3"),
        ("action", "goto:4,5,6"),
    )


def test_segments_are_added_when_quality_control_gave_none() -> None:
    """The source holds one segment, which the link should show."""
    reference = _reference(("skeleton", CCF_SOURCE))
    assert resolve_reference(reference, ASSET_URI, {CCF_SOURCE}) == _reference(
        ("skeleton", f"{ASSET_URI}/{CCF_SOURCE}"), ("segments", "1")
    )


def test_a_dry_run_leaves_the_reference_relative() -> None:
    """Without a destination there is no URI to resolve against."""
    reference = _reference(("skeleton", CCF_SOURCE), ("segments", "1"))
    assert resolve_reference(reference, None, {CCF_SOURCE}) == reference


@pytest.mark.parametrize("asset_uri", [ASSET_URI, None])
def test_a_skeleton_the_asset_lacks_is_removed(asset_uri: str | None) -> None:
    """A link to a source that was not written is dropped, keeping the rest.

    Parameters
    ----------
    asset_uri : str | None
        Asset URI, or a dry run; neither may publish a dangling link.
    """
    reference = _reference(
        ("image", IMAGE), ("skeleton", CCF_SOURCE), ("segments", "1"), ("action", "goto:1,2,3")
    )
    assert resolve_reference(reference, asset_uri, set()) == _reference(
        ("image", IMAGE), ("action", "goto:1,2,3")
    )


def test_a_reference_left_empty_becomes_none() -> None:
    """A reference that held only the missing skeleton has nothing left to open."""
    reference = _reference(("skeleton", CCF_SOURCE), ("segments", "1"))
    assert resolve_reference(reference, ASSET_URI, set()) is None


@pytest.mark.parametrize(
    "reference",
    [
        None,
        "https://example.org/plot.png",
        _reference(("skeleton", "s3://bucket/qc/run/skeletons"), ("segments", "3")),
        _reference(("image", IMAGE), ("action", "goto:1,2,3")),
    ],
)
def test_other_references_are_left_alone(reference: str | None) -> None:
    """Absolute sources, other kinds of reference and no reference are not touched.

    Parameters
    ----------
    reference : str | None
        A reference with nothing to resolve.
    """
    assert resolve_reference(reference, ASSET_URI, {CCF_SOURCE}) == reference


def _with_references(*references: str | None) -> QualityControl:
    """Build a record from the fixture with the given references on its first metrics.

    Parameters
    ----------
    *references : str | None
        References to set, one per metric from the first.

    Returns
    -------
    QualityControl
        The record.
    """
    payload = _payload()
    for metric, reference in zip(payload["metrics"], references):
        metric["reference"] = reference
    return QualityControl.model_validate(payload)


def _cell_dir(tmp_path: Path, *sources: str) -> Path:
    """Create a cell directory holding the given precomputed sources.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.
    *sources : str
        Asset-relative source directories to create.

    Returns
    -------
    Path
        The cell directory.
    """
    cell_dir = tmp_path / "exaSPIM_794492_x_reconstruction-N001_y"
    cell_dir.mkdir()
    for source in sources:
        (cell_dir / source).mkdir(parents=True)
    return cell_dir


def test_a_record_is_resolved_against_its_cell(tmp_path: Path) -> None:
    """Each space's references point at that space's source; status is unchanged."""
    record = _with_references(
        _reference(("skeleton", CCF_SOURCE), ("segments", "1")),
        _reference(("skeleton", SPECIMEN_SOURCE), ("segments", "1")),
    )
    cell_dir = _cell_dir(tmp_path, CCF_SOURCE, SPECIMEN_SOURCE)
    resolved = resolve_skeleton_references(record, ASSET_URI, cell_dir)

    first, second = resolved.metrics[:2]
    assert f"skeleton={ASSET_URI}/{CCF_SOURCE}&segments=1" in first.reference
    assert f"skeleton={ASSET_URI}/{SPECIMEN_SOURCE}&segments=1" in second.reference
    assert resolved.metrics[2:] == record.metrics[2:]
    assert resolved.status == record.status
    assert QualityControl.model_validate_json(resolved.model_dump_json()) == resolved


def test_a_record_with_nothing_to_resolve_is_returned_as_is(tmp_path: Path) -> None:
    """No copy is made when no reference changes."""
    record = _with_references(_reference(("skeleton", CCF_SOURCE)))
    cell_dir = _cell_dir(tmp_path, CCF_SOURCE)
    assert resolve_skeleton_references(record, None, cell_dir) is record
    plain = QualityControl.model_validate(_payload())
    assert resolve_skeleton_references(plain, ASSET_URI, cell_dir) is plain


def test_a_missing_source_is_reported_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every metric of a stage shares its source, so one warning covers them all."""
    reference = _reference(("skeleton", SPECIMEN_SOURCE), ("action", "goto:1,2,3"))
    record = _with_references(reference, reference, reference)
    cell_dir = _cell_dir(tmp_path, CCF_SOURCE)
    with caplog.at_level(logging.WARNING):
        resolved = resolve_skeleton_references(record, ASSET_URI, cell_dir)
    assert caplog.text.count(f"has no {SPECIMEN_SOURCE}") == 1
    expected = _reference(("action", "goto:1,2,3"))
    assert all(metric.reference == expected for metric in resolved.metrics[:3])


def test_a_dry_run_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A dry run says why its references are still relative."""
    record = _with_references(_reference(("skeleton", CCF_SOURCE)))
    with caplog.at_level(logging.INFO):
        resolve_skeleton_references(record, None, _cell_dir(tmp_path, CCF_SOURCE))
    assert "Dry run: quality control references" in caplog.text
