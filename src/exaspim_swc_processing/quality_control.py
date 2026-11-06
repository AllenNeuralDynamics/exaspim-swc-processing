"""Find each reconstruction's quality control record among the pipeline outputs.

Quality control is evaluated per reconstruction upstream and arrives as an AIND
:class:`~aind_data_schema.core.quality_control.QualityControl` per cell. Packaging carries
that record into the cell's derived asset as ``quality_control.json``, so this module finds
the records and keys them by :class:`~exaspim_swc_processing.naming.ReconstructionId`.

A record is matched to its cell by content, not by where it was found: every metric carries
the reconstruction stem in its ``swc`` tag. The file name, then the parent directory name,
are used only for metrics without that tag. A file that turns out to cover several cells is
split, so no cell is published with another cell's metrics.

Quality control is optional. A file that cannot be read or matched is logged and skipped
rather than failing the run, and a cell without a record is still packaged.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aind_data_schema.core.quality_control import CurationMetric, QCMetric, QualityControl

from exaspim_swc_processing.naming import ReconstructionId, ReconstructionNameError, parse_stem

logger = logging.getLogger(__name__)

QUALITY_CONTROL_FILENAME = "quality_control.json"
"""Name the record is written under in each cell directory, per the AIND file set."""

QUALITY_CONTROL_GLOB = "*quality_control*.json"
"""Pattern matching quality control files anywhere under the stage-output root.

Matches ``quality_control.json`` as well as per-cell names such as
``N001-123456-HP_quality_control.json``.
"""

CELL_TAG = "swc"
"""Metric tag holding the reconstruction stem the metric was evaluated on."""

_FILENAME_SUFFIX = "_quality_control"


def _parse_or_none(stem: str) -> ReconstructionId | None:
    """Parse a reconstruction stem, returning ``None`` if it is not one.

    Parameters
    ----------
    stem : str
        Candidate reconstruction stem.

    Returns
    -------
    ReconstructionId | None
        The parsed identifier, or ``None`` if ``stem`` does not follow the convention.
    """
    try:
        return parse_stem(stem)
    except ReconstructionNameError:
        return None


def cell_from_path(path: Path) -> ReconstructionId | None:
    """Infer the reconstruction a quality control file belongs to from where it is.

    Tries the file stem with any ``_quality_control`` suffix removed, then the name of the
    directory holding the file.

    Parameters
    ----------
    path : Path
        Path of the quality control file.

    Returns
    -------
    ReconstructionId | None
        The reconstruction, or ``None`` if neither name is a reconstruction stem.
    """
    for candidate in (path.stem.removesuffix(_FILENAME_SUFFIX), path.parent.name):
        reconstruction = _parse_or_none(candidate)
        if reconstruction is not None:
            return reconstruction
    return None


def split_by_cell(record: QualityControl, path: Path) -> dict[ReconstructionId, QualityControl]:
    """Divide a quality control record into one record per reconstruction.

    Metrics are grouped by their ``swc`` tag. Metrics without the tag are assigned to the
    reconstruction named by the file's location; see :func:`cell_from_path`. Metrics that
    cannot be assigned are logged and dropped.

    Parameters
    ----------
    record : QualityControl
        The record as read.
    path : Path
        Where it was read from, used to place untagged metrics and in log messages.

    Returns
    -------
    dict[ReconstructionId, QualityControl]
        One record per reconstruction. A record covering a single reconstruction is returned
        unchanged; split records keep the original's top-level fields and have their
        ``status`` recomputed from their own metrics.
    """
    groups: dict[ReconstructionId, list[QCMetric | CurationMetric]] = {}
    from_path = cell_from_path(path)
    for metric in record.metrics:
        stem = metric.tags.get(CELL_TAG)
        reconstruction = from_path if stem is None else _parse_or_none(stem)
        if reconstruction is None:
            logger.warning(
                "Dropping quality control metric %r from %s: cannot tell which "
                "reconstruction it belongs to (%s tag %r)",
                metric.name,
                path,
                CELL_TAG,
                stem,
            )
            continue
        groups.setdefault(reconstruction, []).append(metric)

    if len(groups) == 1 and sum(map(len, groups.values())) == len(record.metrics):
        return {next(iter(groups)): record}
    # Revalidated rather than copied, so each part's status reflects only its own metrics.
    base = record.model_dump(exclude={"metrics", "status"})
    return {
        reconstruction: QualityControl.model_validate({
            **base,
            "metrics": [metric.model_dump() for metric in metrics],
        })
        for reconstruction, metrics in groups.items()
    }


def discover_quality_control(stage_root: Path) -> dict[ReconstructionId, QualityControl]:
    """Find the quality control record of every reconstruction under ``stage_root``.

    Parameters
    ----------
    stage_root : Path
        Root directory holding the pipeline stage outputs. Searched recursively for
        :data:`QUALITY_CONTROL_GLOB`.

    Returns
    -------
    dict[ReconstructionId, QualityControl]
        Records keyed by reconstruction, ordered by stem. Records found for the same
        reconstruction in more than one file are combined.
    """
    found: dict[ReconstructionId, QualityControl] = {}
    for path in sorted(stage_root.rglob(QUALITY_CONTROL_GLOB)):
        if not path.is_file():
            continue
        try:
            record = QualityControl.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as error:
            logger.warning("Skipping %s: not a valid quality control record: %s", path, error)
            continue
        for reconstruction, part in split_by_cell(record, path).items():
            if reconstruction not in found:
                found[reconstruction] = part
                continue
            logger.warning(
                "Combining quality control for %s found in more than one file, including %s",
                reconstruction.stem,
                path,
            )
            found[reconstruction] = found[reconstruction] + part
    return dict(sorted(found.items(), key=lambda item: item[0].stem))
