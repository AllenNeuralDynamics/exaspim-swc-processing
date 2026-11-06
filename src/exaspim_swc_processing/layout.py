"""Regroup pipeline stage outputs into one directory per reconstruction.

The pipeline stages each write a flat directory holding every neuron of a subject, so the
outputs for a single cell are scattered across several directories that share only the file
stem. This module inverts that: it indexes the stage outputs by
:class:`~exaspim_swc_processing.naming.ReconstructionId` and materialises one self-contained
directory per cell, laid out as an AIND derived data asset.

Specimen-space reconstructions are published in physical coordinates (microns) at two
densities: ``refined`` is the full-resolution tracing and ``resampled`` is the sparser form
produced alongside the CCF resampling. The upstream ``final-voxel`` outputs are not carried.

``specimen_resampled`` is optional.

Stage outputs are never modified; artifacts are hardlinked where the filesystem allows it and
copied otherwise, so a run costs little beyond the directory entries.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from exaspim_swc_processing.naming import ReconstructionId, ReconstructionNameError, parse_stem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactSpec:
    """Describes one artifact that a cell directory collects.

    Attributes
    ----------
    role : str
        Stable key for this artifact, used in reports and error messages.
    sources : tuple[PurePosixPath, ...]
        Candidate directories relative to the stage-output root, in priority order. The first
        one that exists is used, which absorbs layout drift between upstream asset versions.
    destination : PurePosixPath
        Directory relative to the cell directory that the artifact is placed in.
    suffix : str
        File extension the artifact is found and written with, including the leading dot.
    required : bool
        Whether a cell is considered incomplete when this artifact is absent.
    """

    role: str
    sources: tuple[PurePosixPath, ...]
    destination: PurePosixPath
    suffix: str
    required: bool


def _spec(
    role: str,
    sources: Iterable[str],
    destination: str,
    suffix: str = ".swc",
    *,
    required: bool = True,
) -> ArtifactSpec:
    """Build an :class:`ArtifactSpec` from string paths.

    Parameters
    ----------
    role : str
        Stable key for the artifact.
    sources : Iterable[str]
        Candidate source directories relative to the stage-output root, in priority order.
    destination : str
        Destination directory relative to the cell directory.
    suffix : str, optional
        File extension including the leading dot, by default ``".swc"``.
    required : bool, optional
        Whether absence makes a cell incomplete, by default ``True``.

    Returns
    -------
    ArtifactSpec
        The constructed specification.
    """
    return ArtifactSpec(
        role=role,
        sources=tuple(PurePosixPath(source) for source in sources),
        destination=PurePosixPath(destination),
        suffix=suffix,
        required=required,
    )


ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    _spec(
        "specimen_refined",
        ("refinement/final-world", "swc_refinement/final-world", "final-world"),
        "specimen_space_reconstructions/refined",
    ),
    _spec(
        "specimen_resampled",
        ("refinement/final-world-resampled",),
        "specimen_space_reconstructions/resampled",
        required=False,
    ),
    _spec(
        "ccf",
        ("final/ccf_space_reconstructions/swcs", "ccf_space_reconstructions/swcs"),
        "ccf_space_reconstructions",
    ),
)
"""Artifacts collected into each cell directory, in the order they are placed."""


@dataclass
class CellArtifacts:
    """The artifacts discovered for a single reconstruction.

    Attributes
    ----------
    reconstruction : ReconstructionId
        Identifier of the cell.
    artifacts : dict[str, Path]
        Source path of each discovered artifact, keyed by :attr:`ArtifactSpec.role`.
    """

    reconstruction: ReconstructionId
    artifacts: dict[str, Path] = field(default_factory=dict)

    def missing_roles(self, specs: Iterable[ArtifactSpec] = ARTIFACT_SPECS) -> tuple[str, ...]:
        """Roles that are required by ``specs`` but absent from this cell.

        Parameters
        ----------
        specs : Iterable[ArtifactSpec], optional
            Specifications to check against, by default :data:`ARTIFACT_SPECS`.

        Returns
        -------
        tuple[str, ...]
            Required roles with no discovered artifact, in specification order.
        """
        return tuple(
            spec.role for spec in specs if spec.required and spec.role not in self.artifacts
        )


def resolve_source_dir(stage_root: Path, spec: ArtifactSpec) -> Path | None:
    """Find the directory a spec's artifacts live in under ``stage_root``.

    Parameters
    ----------
    stage_root : Path
        Root directory holding the pipeline stage outputs.
    spec : ArtifactSpec
        Specification whose candidate sources are searched, in priority order.

    Returns
    -------
    Path | None
        The first candidate that exists as a directory, or ``None`` if none do.
    """
    for candidate in spec.sources:
        resolved = stage_root / candidate
        if resolved.is_dir():
            return resolved
    return None


def _iter_artifact_files(source_dir: Path, suffix: str) -> Iterator[Path]:
    """Yield the files in ``source_dir`` carrying ``suffix``, in a stable order.

    Parameters
    ----------
    source_dir : Path
        Directory to scan, non-recursively.
    suffix : str
        File extension including the leading dot.

    Yields
    ------
    Path
        Each matching file, sorted by name.
    """
    yield from sorted(path for path in source_dir.iterdir() if path.suffix == suffix)


def discover_cells(
    stage_root: Path,
    specs: Iterable[ArtifactSpec] = ARTIFACT_SPECS,
) -> dict[ReconstructionId, CellArtifacts]:
    """Index pipeline stage outputs by reconstruction.

    Files whose stem does not follow the reconstruction naming convention are logged and
    skipped rather than aborting the scan, so one stray file cannot fail a whole run.

    Parameters
    ----------
    stage_root : Path
        Root directory holding the pipeline stage outputs.
    specs : Iterable[ArtifactSpec], optional
        Artifacts to look for, by default :data:`ARTIFACT_SPECS`.

    Returns
    -------
    dict[ReconstructionId, CellArtifacts]
        Discovered cells keyed by identifier, ordered by stem.
    """
    specs = tuple(specs)
    discovered: dict[ReconstructionId, CellArtifacts] = {}
    for spec in specs:
        source_dir = resolve_source_dir(stage_root, spec)
        if source_dir is None:
            logger.warning(
                "No source directory found for role %r under %s; tried %s",
                spec.role,
                stage_root,
                ", ".join(str(candidate) for candidate in spec.sources),
            )
            continue
        for path in _iter_artifact_files(source_dir, spec.suffix):
            try:
                reconstruction = parse_stem(path.stem)
            except ReconstructionNameError as error:
                logger.warning("Skipping %s: %s", path, error)
                continue
            cell = discovered.setdefault(reconstruction, CellArtifacts(reconstruction))
            cell.artifacts[spec.role] = path
    return dict(sorted(discovered.items(), key=lambda item: item[0].stem))


def place_artifact(source: Path, destination: Path) -> None:
    """Materialise ``source`` at ``destination``, preferring a hardlink over a copy.

    Hardlinking keeps a run's disk cost near zero; it fails across filesystems and on some
    network mounts, where a copy is used instead.

    Parameters
    ----------
    source : Path
        Existing file to place.
    destination : Path
        Path to create. Parent directories are created as needed; an existing file is replaced.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_cell_directory(
    cell: CellArtifacts,
    cell_dir: Path,
    specs: Iterable[ArtifactSpec] = ARTIFACT_SPECS,
) -> dict[str, Path]:
    """Write one cell's artifacts into ``cell_dir``.

    Parameters
    ----------
    cell : CellArtifacts
        The cell and its discovered source artifacts.
    cell_dir : Path
        Directory to populate. Created if absent.
    specs : Iterable[ArtifactSpec], optional
        Specifications describing where each role is placed, by default :data:`ARTIFACT_SPECS`.

    Returns
    -------
    dict[str, Path]
        Written path of each artifact, keyed by role.
    """
    written: dict[str, Path] = {}
    for spec in specs:
        source = cell.artifacts.get(spec.role)
        if source is None:
            continue
        destination = cell_dir / spec.destination / f"{cell.reconstruction.stem}{spec.suffix}"
        place_artifact(source, destination)
        written[spec.role] = destination
    return written


def build_cell_layout(
    stage_root: Path,
    output_root: Path,
    cell_dir_names: Mapping[ReconstructionId, str],
    specs: Iterable[ArtifactSpec] = ARTIFACT_SPECS,
) -> dict[ReconstructionId, Path]:
    """Regroup every discovered cell under ``output_root``.

    Parameters
    ----------
    stage_root : Path
        Root directory holding the pipeline stage outputs.
    output_root : Path
        Directory the per-cell directories are created in.
    cell_dir_names : Mapping[ReconstructionId, str]
        Directory name to use for each cell, typically the AIND derived asset name from
        :func:`~exaspim_swc_processing.naming.derived_asset_name`.
    specs : Iterable[ArtifactSpec], optional
        Artifacts to collect, by default :data:`ARTIFACT_SPECS`.

    Returns
    -------
    dict[ReconstructionId, Path]
        The directory written for each cell.

    Raises
    ------
    KeyError
        If a discovered cell has no entry in ``cell_dir_names``.
    """
    specs = tuple(specs)
    cells = discover_cells(stage_root, specs)
    written: dict[ReconstructionId, Path] = {}
    for reconstruction, cell in cells.items():
        missing = cell.missing_roles(specs)
        if missing:
            logger.warning(
                "Cell %s is missing required artifact(s): %s",
                reconstruction.stem,
                ", ".join(missing),
            )
        if reconstruction not in cell_dir_names:
            raise KeyError(f"No directory name supplied for reconstruction {reconstruction.stem!r}")
        cell_dir = output_root / cell_dir_names[reconstruction]
        write_cell_directory(cell, cell_dir, specs)
        written[reconstruction] = cell_dir
    logger.info("Wrote %d cell director%s", len(written), "y" if len(written) == 1 else "ies")
    return written
