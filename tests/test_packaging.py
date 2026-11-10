"""Tests for :mod:`exaspim_swc_processing.packaging`."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.metadata import Metadata
from aind_data_schema.core.processing import DataProcess, Processing
from aind_data_schema.core.quality_control import QualityControl

from exaspim_swc_processing.naming import parse_stem
from exaspim_swc_processing.packaging import (
    DATA_DESCRIPTION_FILENAME,
    PROCESSING_FILENAME,
    package_cells,
)
from exaspim_swc_processing.parent_metadata import MetadataSource, ParentMetadata
from exaspim_swc_processing.quality_control import QUALITY_CONTROL_FILENAME

RESOURCES = Path(__file__).parent / "resources"
STEMS = ("N001-794492-HP", "N003-794492-JG")
STAGE_FILES = {
    "refinement/final-world": ".swc",
    "refinement/final-world-resampled": ".swc",
    "final/ccf_space_reconstructions/swcs": ".swc",
    "refinement/raw": ".swc",
}
CREATION_TIME = datetime(2026, 8, 19, 22, 10, 2, tzinfo=timezone.utc)
CCF_SOURCE = "ccf_space_reconstructions/reconstructions.precomputed"
SPECIMEN_SOURCE = "specimen_space_reconstructions/reconstructions.precomputed"
RAW_SOURCE = "raw_reconstructions/reconstructions.precomputed"
PIPELINE = Code(
    url="https://github.com/peter-grotz/exaspim-swc-processing-pipeline",
    name="exaspim-swc-processing",
    version="0.1.0",
)
PACKAGER = Code(
    url="https://github.com/AllenNeuralDynamics/exaspim-swc-processing",
    name="exaspim-swc-processing",
    version="0.1.0",
)


@pytest.fixture
def stage_root(tmp_path: Path) -> Path:
    """Build a stage-output tree holding two complete cells.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    Path
        Root of the generated tree.
    """
    root = tmp_path / "stages"
    swc = (RESOURCES / "reconstruction.swc").read_text(encoding="utf-8")
    for relative, suffix in STAGE_FILES.items():
        directory = root / relative
        directory.mkdir(parents=True)
        for stem in STEMS:
            (directory / f"{stem}{suffix}").write_text(swc, encoding="utf-8")
    return root


def _parent(source: MetadataSource = MetadataSource.S3, upgraded: bool = True) -> ParentMetadata:
    """Build resolved parent metadata from the real fixture.

    Parameters
    ----------
    source : MetadataSource, optional
        Which source it came from, by default S3.
    upgraded : bool, optional
        Whether it was upgraded locally, by default True.

    Returns
    -------
    ParentMetadata
        The resolved parent.
    """
    payload = json.loads((RESOURCES / "parent_data_description.json").read_text(encoding="utf-8"))
    return ParentMetadata(
        data_description=DataDescription.model_validate(payload),
        asset_name=payload["name"],
        source=source,
        source_schema_version="1.0.4" if upgraded else "2.4.1",
        upgraded=upgraded,
    )


def _stage_processes() -> list[DataProcess]:
    """Load the pipeline's stage records from a real run.

    Returns
    -------
    list[DataProcess]
        The records in execution order.
    """
    payload = json.loads((RESOURCES / "stage_data_processes.json").read_text(encoding="utf-8"))
    return [DataProcess.model_validate(record) for record in payload]


def _package(stage_root: Path, tmp_path: Path, **kwargs: object) -> object:
    """Package the fixture tree with default arguments.

    Parameters
    ----------
    stage_root : Path
        Stage output tree.
    tmp_path : Path
        Temporary directory.
    **kwargs : object
        Overrides passed to :func:`package_cells`.

    Returns
    -------
    object
        The packaging result.
    """
    parent = kwargs.pop("parent", None) or _parent()
    return package_cells(
        stage_root,
        tmp_path / "out",
        parent,
        _stage_processes(),
        PIPELINE,
        CREATION_TIME,
        **kwargs,
    )


def test_each_cell_gets_its_own_directory(stage_root: Path, tmp_path: Path) -> None:
    """One derived asset directory per reconstruction."""
    result = _package(stage_root, tmp_path)
    assert len(result.packaged) == len(STEMS)
    assert result.skipped == ()
    names = {cell.asset_name for cell in result.packaged}
    assert names == {
        "exaSPIM_794492_2026-01-09_16-50-40_reconstruction-N001_2026-08-19_22-10-02",
        "exaSPIM_794492_2026-01-09_16-50-40_reconstruction-N003_2026-08-19_22-10-02",
    }


def test_exactly_two_metadata_files_are_written(stage_root: Path, tmp_path: Path) -> None:
    """Without QC, only data_description and processing; no subject or procedures."""
    result = _package(stage_root, tmp_path)
    directory = result.packaged[0].directory
    written = {path.name for path in directory.iterdir() if path.is_file()}
    assert written == {DATA_DESCRIPTION_FILENAME, PROCESSING_FILENAME}


def test_written_asset_validates_as_metadata(stage_root: Path, tmp_path: Path) -> None:
    """The two files together form a valid derived asset record."""
    result = _package(stage_root, tmp_path)
    directory = result.packaged[0].directory
    description = DataDescription.model_validate_json(
        (directory / DATA_DESCRIPTION_FILENAME).read_text(encoding="utf-8")
    )
    processing = Processing.model_validate_json(
        (directory / PROCESSING_FILENAME).read_text(encoding="utf-8")
    )
    record = Metadata(
        name=description.name,
        location=f"s3://aind-open-data/{description.name}",
        data_description=description,
        processing=processing,
    )
    assert record.data_description is not None
    assert record.processing is not None


def test_fallback_source_is_tagged_for_later_rederivation(stage_root: Path, tmp_path: Path) -> None:
    """Tags are indexed, so fallback-derived assets can be found and rebuilt."""
    result = _package(stage_root, tmp_path)
    description = DataDescription.model_validate_json(
        (result.packaged[0].directory / DATA_DESCRIPTION_FILENAME).read_text(encoding="utf-8")
    )
    assert description.tags == ["metadata-source:s3", "metadata-upgraded"]


def test_reconstructions_are_written_alongside_the_metadata(
    stage_root: Path, tmp_path: Path
) -> None:
    """The asset holds the data as well as the description of it, named without initials."""
    result = _package(stage_root, tmp_path)
    cell = result.packaged[0]
    entries = {
        path.relative_to(cell.directory).as_posix()
        for path in cell.directory.rglob("*")
        if path.suffix == ".swc" or path.name == "reconstructions.precomputed"
    }
    assert entries == {
        "ccf_space_reconstructions/N001-794492.swc",
        CCF_SOURCE,
        "specimen_space_reconstructions/N001-794492.swc",
        "specimen_space_reconstructions/N001-794492_dense.swc",
        SPECIMEN_SOURCE,
        "raw_reconstructions/N001-794492.swc",
        RAW_SOURCE,
    }


def test_a_cell_whose_description_cannot_be_derived_is_skipped(
    stage_root: Path, tmp_path: Path
) -> None:
    """A cell is skipped rather than published without metadata."""
    parent = _parent()
    broken = ParentMetadata(
        data_description=parent.data_description.model_copy(update={"name": "not-an-asset-name"}),
        asset_name="not-an-asset-name",
        source=parent.source,
        source_schema_version=parent.source_schema_version,
        upgraded=parent.upgraded,
    )
    result = _package(stage_root, tmp_path, parent=broken)
    assert result.packaged == ()
    assert len(result.skipped) == len(STEMS)
    assert "not-an-asset-name" in result.skipped[0].reason


def test_an_incomplete_cell_is_still_packaged(stage_root: Path, tmp_path: Path) -> None:
    """A missing required artifact is logged, not fatal; the transform skips failures."""
    (stage_root / f"final/ccf_space_reconstructions/swcs/{STEMS[0]}.swc").unlink()
    result = _package(stage_root, tmp_path)
    assert len(result.packaged) == len(STEMS)


def test_overrides_apply_to_every_cell(stage_root: Path, tmp_path: Path) -> None:
    """Fields supplied as configuration reach all published descriptions."""
    result = _package(stage_root, tmp_path, overrides={"project_name": "Neuron Reconstruction"})
    for cell in result.packaged:
        description = DataDescription.model_validate_json(
            (cell.directory / DATA_DESCRIPTION_FILENAME).read_text(encoding="utf-8")
        )
        assert description.project_name == "Neuron Reconstruction"


def test_packaging_does_not_record_itself_as_a_process(stage_root: Path, tmp_path: Path) -> None:
    """Hardlinking files into a directory produces no data, so it is not a process."""
    result = _package(stage_root, tmp_path)
    processing = Processing.model_validate_json(
        (result.packaged[0].directory / PROCESSING_FILENAME).read_text(encoding="utf-8")
    )
    names = [process.name for process in processing.data_processes]
    assert names == [process.name for process in _stage_processes()]
    assert "exaspim_swc_packaging" not in names
    assert set(processing.dependency_graph) == set(names)


def test_the_metadata_source_stays_discoverable_on_tags(stage_root: Path, tmp_path: Path) -> None:
    """Dropping the process loses output_parameters; tags carry the source and index."""
    result = _package(stage_root, tmp_path)
    description = json.loads(
        (result.packaged[0].directory / DATA_DESCRIPTION_FILENAME).read_text(encoding="utf-8")
    )
    assert any(tag.startswith("metadata-source:") for tag in description["tags"])


def _quality_control() -> QualityControl:
    """Load the quality control fixture, which covers ``STEMS[0]``.

    Returns
    -------
    QualityControl
        The record.
    """
    return QualityControl.model_validate_json(
        (RESOURCES / "quality_control.json").read_text(encoding="utf-8")
    )


def test_quality_control_is_written_for_the_cells_that_have_it(
    stage_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cell with a record carries it; one without is still packaged, with a warning."""
    record = _quality_control()
    with caplog.at_level(logging.WARNING):
        result = _package(stage_root, tmp_path, quality_control={parse_stem(STEMS[0]): record})
    with_qc, without_qc = result.packaged
    assert with_qc.has_quality_control
    assert not without_qc.has_quality_control
    assert not (without_qc.directory / QUALITY_CONTROL_FILENAME).exists()
    assert f"No quality control for {STEMS[1]}" in caplog.text

    written = QualityControl.model_validate_json(
        (with_qc.directory / QUALITY_CONTROL_FILENAME).read_text(encoding="utf-8")
    )
    assert written == record


def test_packaged_quality_control_validates_with_the_asset(
    stage_root: Path, tmp_path: Path
) -> None:
    """The three files together form a valid derived asset record."""
    result = _package(
        stage_root, tmp_path, quality_control={parse_stem(STEMS[0]): _quality_control()}
    )
    directory = result.packaged[0].directory
    description = DataDescription.model_validate_json(
        (directory / DATA_DESCRIPTION_FILENAME).read_text(encoding="utf-8")
    )
    record = Metadata(
        name=description.name,
        location=f"s3://aind-open-data/{description.name}",
        data_description=description,
        processing=Processing.model_validate_json(
            (directory / PROCESSING_FILENAME).read_text(encoding="utf-8")
        ),
        quality_control=QualityControl.model_validate_json(
            (directory / QUALITY_CONTROL_FILENAME).read_text(encoding="utf-8")
        ),
    )
    assert record.quality_control is not None


def test_quality_control_for_an_unknown_cell_is_reported(
    stage_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A stem mismatch between QC and the reconstructions is surfaced, not hidden."""
    with caplog.at_level(logging.WARNING):
        result = _package(
            stage_root,
            tmp_path,
            quality_control={parse_stem("N999-794492-HP"): _quality_control()},
        )
    assert not any(cell.has_quality_control for cell in result.packaged)
    assert "Quality control for N999-794492-HP matches no reconstruction" in caplog.text


def test_each_cell_gets_a_precomputed_source_per_space(stage_root: Path, tmp_path: Path) -> None:
    """Every reconstruction folder can be opened in Neuroglancer from the asset."""
    result = _package(stage_root, tmp_path)
    for cell in result.packaged:
        assert cell.precomputed == ("ccf", "specimen_resampled", "raw")
        for source in (CCF_SOURCE, SPECIMEN_SOURCE, RAW_SOURCE):
            assert (cell.directory / source / "info").is_file()
            assert (cell.directory / source / "1").is_file()


def test_segments_are_labelled_without_initials(stage_root: Path, tmp_path: Path) -> None:
    """The label Neuroglancer shows matches the packaged file names."""
    cell = _package(stage_root, tmp_path).packaged[0]
    properties = json.loads(
        (cell.directory / CCF_SOURCE / "segment_properties" / "info").read_text(encoding="utf-8")
    )
    label = next(p for p in properties["inline"]["properties"] if p["id"] == "label")
    assert label["values"] == ["N001-794492"]


def test_a_cell_without_a_resampled_tracing_has_no_specimen_source(
    stage_root: Path, tmp_path: Path
) -> None:
    """The resampled specimen tracing is optional; its source is skipped without it."""
    (stage_root / f"refinement/final-world-resampled/{STEMS[0]}.swc").unlink()
    result = _package(stage_root, tmp_path)
    assert result.packaged[0].precomputed == ("ccf", "raw")
    assert not (result.packaged[0].directory / SPECIMEN_SOURCE).exists()


def _linked_quality_control() -> QualityControl:
    """Load the QC fixture with its first three metrics linking the CCF, specimen and raw sources.

    Returns
    -------
    QualityControl
        The record, with references as the QC capsule writes them before packaging.
    """
    payload = json.loads((RESOURCES / "quality_control.json").read_text(encoding="utf-8"))
    payload["metrics"][0]["reference"] = (
        f"js:neuroglancer?skeleton={CCF_SOURCE}&segments=1&action=goto:5000,3000,4000"
    )
    payload["metrics"][1]["reference"] = f"js:neuroglancer?skeleton={SPECIMEN_SOURCE}&segments=1"
    payload["metrics"][2]["reference"] = f"js:neuroglancer?skeleton={RAW_SOURCE}&segments=1"
    return QualityControl.model_validate(payload)


@pytest.mark.parametrize("destination", ["s3://aind-open-data/", "s3://bucket/run-1"])
def test_quality_control_links_the_published_sources(
    stage_root: Path, tmp_path: Path, destination: str
) -> None:
    """References point at where the upload will put each source.

    Parameters
    ----------
    stage_root : Path
        Stage output tree.
    tmp_path : Path
        Temporary directory.
    destination : str
        Upload destination.
    """
    result = _package(
        stage_root,
        tmp_path,
        quality_control={parse_stem(STEMS[0]): _linked_quality_control()},
        destination=destination,
    )
    cell = result.packaged[0]
    written = QualityControl.model_validate_json(
        (cell.directory / QUALITY_CONTROL_FILENAME).read_text(encoding="utf-8")
    )
    asset = f"{destination.rstrip('/')}/{cell.asset_name}"
    assert written.metrics[0].reference == (
        f"js:neuroglancer?skeleton={asset}/{CCF_SOURCE}&segments=1&action=goto:5000,3000,4000"
    )
    assert written.metrics[1].reference == (
        f"js:neuroglancer?skeleton={asset}/{SPECIMEN_SOURCE}&segments=1"
    )
    assert written.metrics[2].reference == (
        f"js:neuroglancer?skeleton={asset}/{RAW_SOURCE}&segments=1"
    )


def test_a_dry_run_keeps_quality_control_links_relative(stage_root: Path, tmp_path: Path) -> None:
    """Without a destination the asset location is unknown, so nothing is resolved."""
    record = _linked_quality_control()
    result = _package(stage_root, tmp_path, quality_control={parse_stem(STEMS[0]): record})
    written = QualityControl.model_validate_json(
        (result.packaged[0].directory / QUALITY_CONTROL_FILENAME).read_text(encoding="utf-8")
    )
    assert written == record


def test_an_invalid_destination_fails_before_anything_is_written(
    stage_root: Path, tmp_path: Path
) -> None:
    """A destination that would produce broken links stops the run up front."""
    with pytest.raises(ValueError, match="s3://bucket/prefix/"):
        _package(stage_root, tmp_path, destination="aind-open-data/")
    assert not (tmp_path / "out").exists()
