"""Tests for :mod:`exaspim_swc_processing.layout`."""

import logging
import os
from pathlib import Path

import pytest

from exaspim_swc_processing.layout import (
    ARTIFACT_SPECS,
    CellArtifacts,
    build_cell_layout,
    discover_cells,
    place_artifact,
    resolve_source_dir,
    write_cell_directory,
)
from exaspim_swc_processing.naming import parse_stem

STEMS = ("N001-794492-HP", "N003-794492-JG")

STAGE_FILES = {
    "refinement/final-world": ".swc",
    "refinement/final-world-resampled": ".swc",
    "final/ccf_space_reconstructions/swcs": ".swc",
}


@pytest.fixture
def stage_root(tmp_path: Path) -> Path:
    """Build a miniature stage-output tree holding two complete cells.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    Path
        Root of the generated stage output tree.
    """
    root = tmp_path / "stages"
    for relative, suffix in STAGE_FILES.items():
        directory = root / relative
        directory.mkdir(parents=True)
        for stem in STEMS:
            (directory / f"{stem}{suffix}").write_text(f"{relative}:{stem}", encoding="utf-8")
    return root


def _names(stems: tuple[str, ...] = STEMS) -> dict:
    """Map each stem's identifier to a directory name.

    Parameters
    ----------
    stems : tuple[str, ...], optional
        Stems to map, by default :data:`STEMS`.

    Returns
    -------
    dict
        Mapping of identifier to directory name.
    """
    return {parse_stem(stem): f"asset_{stem}" for stem in stems}


def test_discover_cells_indexes_every_role(stage_root: Path) -> None:
    """Every configured role is discovered for each cell in a complete tree."""
    cells = discover_cells(stage_root)
    assert [identifier.stem for identifier in cells] == list(STEMS)
    expected_roles = {spec.role for spec in ARTIFACT_SPECS}
    for cell in cells.values():
        assert set(cell.artifacts) == expected_roles
        assert cell.missing_roles() == ()


def test_discover_cells_warns_and_skips_unparseable_names(
    stage_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A stray file that is not a reconstruction is skipped rather than aborting the scan."""
    (stage_root / "refinement/final-world/notes.swc").write_text("x", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cells = discover_cells(stage_root)
    assert [identifier.stem for identifier in cells] == list(STEMS)
    assert "Skipping" in caplog.text


def test_discover_cells_warns_when_a_source_directory_is_absent(
    stage_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A role with no source directory is reported and the remaining roles still index."""
    for path in (stage_root / "refinement/final-world-resampled").iterdir():
        path.unlink()
    (stage_root / "refinement/final-world-resampled").rmdir()
    with caplog.at_level(logging.WARNING):
        cells = discover_cells(stage_root)
    assert "specimen_resampled" in caplog.text
    for cell in cells.values():
        assert "specimen_resampled" not in cell.artifacts


def test_missing_roles_reports_only_required_artifacts(stage_root: Path) -> None:
    """An absent optional artifact does not make a cell incomplete; a required one does."""
    for path in (stage_root / "final/ccf_space_reconstructions/swcs").iterdir():
        path.unlink()
    for path in (stage_root / "refinement/final-world-resampled").iterdir():
        path.unlink()
    cells = discover_cells(stage_root)
    for cell in cells.values():
        assert cell.missing_roles() == ("ccf",)


def test_resolve_source_dir_prefers_the_first_existing_candidate(stage_root: Path) -> None:
    """Candidate source directories are tried in priority order."""
    spec = next(spec for spec in ARTIFACT_SPECS if spec.role == "specimen_refined")
    assert resolve_source_dir(stage_root, spec) == stage_root / "refinement/final-world"


def test_resolve_source_dir_returns_none_when_no_candidate_exists(tmp_path: Path) -> None:
    """An empty tree resolves to no source directory."""
    spec = next(spec for spec in ARTIFACT_SPECS if spec.role == "ccf")
    assert resolve_source_dir(tmp_path, spec) is None


def test_place_artifact_hardlinks_when_possible(tmp_path: Path) -> None:
    """Placing an artifact on the same filesystem creates a hardlink, not a copy."""
    source = tmp_path / "source.swc"
    source.write_text("payload", encoding="utf-8")
    destination = tmp_path / "nested" / "destination.swc"
    place_artifact(source, destination)
    assert destination.read_text(encoding="utf-8") == "payload"
    assert source.stat().st_ino == destination.stat().st_ino


def test_place_artifact_falls_back_to_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When hardlinking is unsupported the artifact is copied instead."""

    def _refuse(source: object, destination: object) -> None:
        """Stand in for :func:`os.link` on a filesystem that cannot hardlink.

        Parameters
        ----------
        source : object
            Ignored.
        destination : object
            Ignored.

        Raises
        ------
        OSError
            Always.
        """
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", _refuse)
    source = tmp_path / "source.swc"
    source.write_text("payload", encoding="utf-8")
    destination = tmp_path / "destination.swc"
    place_artifact(source, destination)
    assert destination.read_text(encoding="utf-8") == "payload"
    assert source.stat().st_ino != destination.stat().st_ino


def test_place_artifact_replaces_an_existing_destination(tmp_path: Path) -> None:
    """Re-placing an artifact overwrites a stale file left by an earlier run."""
    source = tmp_path / "source.swc"
    source.write_text("new", encoding="utf-8")
    destination = tmp_path / "destination.swc"
    destination.write_text("stale", encoding="utf-8")
    place_artifact(source, destination)
    assert destination.read_text(encoding="utf-8") == "new"


def test_write_cell_directory_skips_absent_roles(tmp_path: Path, stage_root: Path) -> None:
    """Only discovered artifacts are written."""
    cell = discover_cells(stage_root)[parse_stem(STEMS[0])]
    del cell.artifacts["specimen_resampled"]
    written = write_cell_directory(cell, tmp_path / "cell")
    assert "specimen_resampled" not in written
    assert set(written) == {spec.role for spec in ARTIFACT_SPECS} - {"specimen_resampled"}


def test_build_cell_layout_writes_the_expected_tree(stage_root: Path, tmp_path: Path) -> None:
    """Each cell directory holds the full per-cell layout."""
    output_root = tmp_path / "out"
    written = build_cell_layout(stage_root, output_root, _names())
    assert len(written) == len(STEMS)
    stem = STEMS[0]
    cell_dir = written[parse_stem(stem)]
    assert cell_dir == output_root / f"asset_{stem}"
    for relative in (
        f"specimen_space_reconstructions/refined/{stem}.swc",
        f"ccf_space_reconstructions/{stem}.swc",
    ):
        assert (cell_dir / relative).is_file(), relative


def test_build_cell_layout_warns_but_continues_on_an_incomplete_cell(
    stage_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cell whose transform failed upstream is reported and still written."""
    (stage_root / f"final/ccf_space_reconstructions/swcs/{STEMS[0]}.swc").unlink()
    with caplog.at_level(logging.WARNING):
        written = build_cell_layout(stage_root, tmp_path / "out", _names())
    assert len(written) == len(STEMS)
    assert "missing required artifact" in caplog.text
    assert STEMS[0] in caplog.text


def test_build_cell_layout_rejects_an_unnamed_cell(stage_root: Path, tmp_path: Path) -> None:
    """A discovered cell with no supplied directory name is an error, not a silent skip."""
    with pytest.raises(KeyError, match=STEMS[1]):
        build_cell_layout(stage_root, tmp_path / "out", _names((STEMS[0],)))


def test_build_cell_layout_logs_singular_for_one_cell(stage_root: Path, tmp_path: Path) -> None:
    """The summary line reads naturally when a single cell is written."""
    for directory, suffix in STAGE_FILES.items():
        (stage_root / directory / f"{STEMS[1]}{suffix}").unlink()
    written = build_cell_layout(stage_root, tmp_path / "out", _names((STEMS[0],)))
    assert len(written) == 1


def test_cell_artifacts_defaults_to_no_artifacts() -> None:
    """A freshly constructed cell reports every required role as missing."""
    cell = CellArtifacts(parse_stem(STEMS[0]))
    assert cell.artifacts == {}
    assert set(cell.missing_roles()) == {spec.role for spec in ARTIFACT_SPECS if spec.required}


def test_specimen_resampled_is_optional_until_the_resample_stage_emits_it(
    stage_root: Path,
) -> None:
    """The resample stage does not yet produce it, so its absence is not fatal."""
    for path in (stage_root / "refinement/final-world-resampled").iterdir():
        path.unlink()
    (stage_root / "refinement/final-world-resampled").rmdir()
    for cell in discover_cells(stage_root).values():
        assert "specimen_resampled" not in cell.artifacts
        assert cell.missing_roles() == ()


def test_both_specimen_densities_are_written(stage_root: Path, tmp_path: Path) -> None:
    """Refined and resampled sit side by side under specimen space."""
    written = build_cell_layout(stage_root, tmp_path / "out", _names())
    cell_dir = written[parse_stem(STEMS[0])]
    assert (cell_dir / f"specimen_space_reconstructions/refined/{STEMS[0]}.swc").is_file()
    assert (cell_dir / f"specimen_space_reconstructions/resampled/{STEMS[0]}.swc").is_file()
