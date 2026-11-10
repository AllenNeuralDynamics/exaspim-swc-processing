"""Write each cell directory as a self-contained AIND derived data asset.

Ties the other modules together: :mod:`layout` regroups the stage outputs by cell,
:mod:`naming` and :mod:`data_description` name and describe each one, and :mod:`processing`
records how it was made. This module drives them and writes the result to disk.

Each cell directory receives two metadata files, ``data_description.json`` and
``processing.json``. That is the complete required set for a ``processing``-type derived
asset: ``REQUIRED_FILE_SETS`` in aind-data-schema only demands ``data_description``
alongside ``processing``. ``subject``, ``procedures``, ``acquisition`` and ``instrument``
are deliberately not inherited — including ``subject`` would pull in a four-file
requirement that the upstream assets cannot currently satisfy, so the asset would advertise
itself as incomplete.

A cell's ``quality_control.json`` is written as well when a record was found for it; see
:mod:`~exaspim_swc_processing.quality_control`. It is optional — no file set requires it —
so a cell without one is still packaged, and a warning is logged.

Each cell also receives a Neuroglancer precomputed skeleton source per reconstruction folder,
converted from its reconstructions; see :mod:`~exaspim_swc_processing.precomputed`. When the
asset's S3 destination is given, the quality control references to those sources, which
upstream can only give relative to the asset, are made absolute before the record is
written.

Packaging does not record itself as a
:class:`~aind_data_schema.core.processing.DataProcess`. Regrouping changes no data -- the
reconstructions are hardlinked, byte for byte -- so a record of it would describe
assembling the asset rather than producing the data, in a document meant for the latter.
The precomputed conversion does produce data; the caller records it by appending
:func:`~exaspim_swc_processing.precomputed.conversion_process` to the stage records.
Where the parent metadata was read from is carried on ``DataDescription.tags``, which is
indexed and therefore searchable, unlike ``output_parameters``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess
from aind_data_schema.core.quality_control import QualityControl

from exaspim_swc_processing.data_description import (
    DataDescriptionDerivationError,
    derive_cell_data_description,
)
from exaspim_swc_processing.layout import ARTIFACT_SPECS, ArtifactSpec, discover_cells
from exaspim_swc_processing.layout import write_cell_directory as _write_cell_directory
from exaspim_swc_processing.naming import ReconstructionId, asset_s3_uri
from exaspim_swc_processing.parent_metadata import ParentMetadata
from exaspim_swc_processing.precomputed import write_cell_precomputed
from exaspim_swc_processing.processing import build_cell_processing
from exaspim_swc_processing.quality_control import (
    QUALITY_CONTROL_FILENAME,
    resolve_skeleton_references,
)

logger = logging.getLogger(__name__)

DATA_DESCRIPTION_FILENAME = "data_description.json"
PROCESSING_FILENAME = "processing.json"


@dataclass(frozen=True)
class PackagedCell:
    """One cell directory that was written.

    Attributes
    ----------
    reconstruction : ReconstructionId
        Identifier of the cell.
    directory : Path
        Directory written.
    asset_name : str
        Name of the derived asset, which is also the directory name.
    has_quality_control : bool
        Whether a ``quality_control.json`` was written for the cell.
    precomputed : tuple[str, ...]
        Roles of the reconstructions converted to precomputed skeleton sources.
    """

    reconstruction: ReconstructionId
    directory: Path
    asset_name: str
    has_quality_control: bool = False
    precomputed: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkippedCell:
    """One cell that could not be packaged.

    Attributes
    ----------
    reconstruction : ReconstructionId
        Identifier of the cell.
    reason : str
        Why it was skipped, for the run summary.
    """

    reconstruction: ReconstructionId
    reason: str


@dataclass(frozen=True)
class PackagingResult:
    """Outcome of packaging a run.

    Attributes
    ----------
    packaged : tuple[PackagedCell, ...]
        Cells written.
    skipped : tuple[SkippedCell, ...]
        Cells not written, with reasons. A skipped cell is distinguishable from a cell
        that produced nothing, which matters when a run reports success.
    """

    packaged: tuple[PackagedCell, ...]
    skipped: tuple[SkippedCell, ...]


def package_cells(
    stage_root: Path,
    output_root: Path,
    parent: ParentMetadata,
    stage_processes: Sequence[DataProcess],
    pipeline: Code,
    creation_time: datetime,
    specs: Sequence[ArtifactSpec] = ARTIFACT_SPECS,
    overrides: Mapping[str, object] | None = None,
    quality_control: Mapping[ReconstructionId, QualityControl] | None = None,
    destination: str | None = None,
) -> PackagingResult:
    """Write every discovered cell as its own derived data asset.

    Parameters
    ----------
    stage_root : Path
        Root holding the pipeline stage outputs.
    output_root : Path
        Directory the cell directories are created in.
    parent : ParentMetadata
        Parent asset metadata and the source it came from.
    stage_processes : Sequence[DataProcess]
        The pipeline's stage records, shared by every cell.
    pipeline : Code
        The pipeline that produced the run.
    creation_time : datetime
        Creation time shared by every cell in the run.
    specs : Sequence[ArtifactSpec], optional
        Artifacts to collect, by default :data:`~exaspim_swc_processing.layout.ARTIFACT_SPECS`.
    overrides : Mapping[str, object] | None, optional
        ``DataDescription`` fields to set explicitly on every cell, overriding the parent.
    quality_control : Mapping[ReconstructionId, QualityControl] | None, optional
        Quality control record of each cell, typically from
        :func:`~exaspim_swc_processing.quality_control.discover_quality_control`. Cells
        without one are packaged without ``quality_control.json``.
    destination : str | None, optional
        ``s3://bucket/prefix/`` the cell directories will be uploaded under, used to make
        the quality control references to each cell's precomputed skeletons absolute.
        ``None`` (a dry run) leaves them relative to the asset. A value that is not an
        ``s3://`` URI with a bucket raises ``ValueError`` before any cell is written.

    Returns
    -------
    PackagingResult
        Cells written and cells skipped. A cell whose description cannot be derived is
        skipped rather than written without metadata.
    """
    asset_s3_uri(destination, "")  # Validate up front, not after the first cell.
    specs = tuple(specs)
    record = build_cell_processing(list(stage_processes), pipeline)
    packaged: list[PackagedCell] = []
    skipped: list[SkippedCell] = []

    quality_control = quality_control or {}
    cells = discover_cells(stage_root, specs)
    for unmatched in sorted(r.stem for r in quality_control if r not in cells):
        logger.warning("Quality control for %s matches no reconstruction", unmatched)

    for reconstruction, cell in cells.items():
        missing = cell.missing_roles(specs)
        if missing:
            logger.warning(
                "Cell %s is missing required artifact(s): %s",
                reconstruction.stem,
                ", ".join(missing),
            )
        try:
            description = derive_cell_data_description(
                parent.data_description,
                reconstruction,
                creation_time,
                tags=parent.tags(),
                **(overrides or {}),
            )
        except DataDescriptionDerivationError as error:
            logger.warning("Skipping %s: %s", reconstruction.stem, error)
            skipped.append(SkippedCell(reconstruction, str(error)))
            continue

        cell_dir = output_root / description.name
        written = _write_cell_directory(cell, cell_dir, specs)
        precomputed = write_cell_precomputed(written, cell_dir, reconstruction.published_stem)
        (cell_dir / DATA_DESCRIPTION_FILENAME).write_text(
            description.model_dump_json(indent=2), encoding="utf-8"
        )
        (cell_dir / PROCESSING_FILENAME).write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )
        qc = quality_control.get(reconstruction)
        if qc is None:
            logger.warning("No quality control for %s", reconstruction.stem)
        else:
            qc = resolve_skeleton_references(
                qc, asset_s3_uri(destination, description.name), cell_dir
            )
            (cell_dir / QUALITY_CONTROL_FILENAME).write_text(
                qc.model_dump_json(indent=2), encoding="utf-8"
            )
        packaged.append(
            PackagedCell(
                reconstruction,
                cell_dir,
                description.name,
                has_quality_control=qc is not None,
                precomputed=tuple(precomputed),
            )
        )

    logger.info("Packaged %d cell(s), skipped %d", len(packaged), len(skipped))
    return PackagingResult(packaged=tuple(packaged), skipped=tuple(skipped))
