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

A metric's ``reference`` may open the cell's precomputed skeleton in Neuroglancer. Quality
control runs before the asset is named, so it can only give the skeleton's path relative to
the asset; :func:`resolve_skeleton_references` makes that path absolute once the asset's S3
location is known.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode

from aind_data_schema.core.quality_control import CurationMetric, QCMetric, QualityControl

from exaspim_swc_processing.naming import ReconstructionId, ReconstructionNameError, parse_stem
from exaspim_swc_processing.precomputed import SEGMENT_ID

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

NEUROGLANCER_REFERENCE_PREFIX = "js:neuroglancer?"
"""Prefix of a metric reference that opens Neuroglancer, as written by ``standard_morph``.

The rest is a query string with ``image``, ``skeleton``, ``segments`` and repeated
``action`` parameters.
"""

SKELETON_PARAMETER = "skeleton"
SEGMENTS_PARAMETER = "segments"
_REFERENCE_SAFE = ":/,"
"""Characters left unescaped when a reference is encoded, matching ``standard_morph``."""


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


def _reference_params(reference: str | None) -> list[tuple[str, str]] | None:
    """Split a Neuroglancer reference into its query parameters.

    Parameters
    ----------
    reference : str | None
        A metric's reference.

    Returns
    -------
    list[tuple[str, str]] | None
        The parameters in order, or ``None`` if ``reference`` is not a
        :data:`NEUROGLANCER_REFERENCE_PREFIX` reference.
    """
    if reference is None or not reference.startswith(NEUROGLANCER_REFERENCE_PREFIX):
        return None
    query = reference.removeprefix(NEUROGLANCER_REFERENCE_PREFIX)
    return parse_qsl(query, keep_blank_values=True)


def _relative_skeleton(params: Sequence[tuple[str, str]]) -> str | None:
    """Return the reference's skeleton source if it is a path relative to the asset.

    Parameters
    ----------
    params : Sequence[tuple[str, str]]
        The reference's query parameters.

    Returns
    -------
    str | None
        The relative path, or ``None`` if there is no skeleton or it is already absolute.
    """
    for key, value in params:
        if key == SKELETON_PARAMETER:
            return None if "://" in value else value
    return None


def resolve_reference(
    reference: str | None,
    asset_uri: str | None,
    available: set[str],
    segment_ids: Sequence[int] = (SEGMENT_ID,),
) -> str | None:
    """Make a Neuroglancer reference's asset-relative skeleton source absolute.

    Parameters
    ----------
    reference : str | None
        A metric's reference. Anything other than a :data:`NEUROGLANCER_REFERENCE_PREFIX`
        reference with a relative ``skeleton`` is returned unchanged.
    asset_uri : str | None
        S3 URI of the asset, or ``None`` for a dry run, where the path is left relative.
    available : set[str]
        Relative skeleton paths that exist in the cell directory. A skeleton outside it is
        removed from the reference, with its ``segments``, rather than left pointing at
        nothing.
    segment_ids : Sequence[int], optional
        Segments to show from a resolved source, by default the one each asset holds.

    Returns
    -------
    str | None
        The reference, or ``None`` if removing the skeleton left it empty.
    """
    params = _reference_params(reference)
    skeleton = None if params is None else _relative_skeleton(params)
    if skeleton is None or (skeleton in available and asset_uri is None):
        return reference

    resolved: list[tuple[str, str]] = []
    for key, value in params:
        if key == SEGMENTS_PARAMETER:
            continue
        if key != SKELETON_PARAMETER:
            resolved.append((key, value))
        elif skeleton in available:
            resolved.append((key, f"{asset_uri}/{skeleton}"))
            resolved.append((SEGMENTS_PARAMETER, ",".join(str(s) for s in segment_ids)))
    if not resolved:
        return None
    query = urlencode(resolved, quote_via=quote, safe=_REFERENCE_SAFE)
    return NEUROGLANCER_REFERENCE_PREFIX + query


def resolve_skeleton_references(
    record: QualityControl,
    asset_uri: str | None,
    cell_dir: Path,
    segment_ids: Sequence[int] = (SEGMENT_ID,),
) -> QualityControl:
    """Point every metric reference at the cell's published precomputed skeletons.

    Quality control names the skeleton by its path relative to the asset, because the asset
    is named only when it is packaged. This replaces each such path with its S3 URI under
    ``asset_uri`` and sets ``segments`` to the ones the source holds. See
    :func:`resolve_reference` for the cases left alone.

    Parameters
    ----------
    record : QualityControl
        The cell's record.
    asset_uri : str | None
        S3 URI the asset is uploaded to, or ``None`` for a dry run.
    cell_dir : Path
        The cell directory, which is checked for each referenced source.
    segment_ids : Sequence[int], optional
        Segments to show, by default the one each asset holds.

    Returns
    -------
    QualityControl
        The record with resolved references, or ``record`` itself if none changed. Statuses
        are unaffected.
    """
    paths = {
        _relative_skeleton(params)
        for params in map(_reference_params, (metric.reference for metric in record.metrics))
        if params is not None
    } - {None}
    available = {path for path in paths if (cell_dir / path).is_dir()}
    for path in sorted(paths - available):
        logger.warning(
            "%s has no %s; removing it from the quality control references", cell_dir.name, path
        )
    if available and asset_uri is None:
        logger.info("Dry run: quality control references for %s stay relative", cell_dir.name)

    references = [
        resolve_reference(metric.reference, asset_uri, available, segment_ids)
        for metric in record.metrics
    ]
    if references == [metric.reference for metric in record.metrics]:
        return record
    metrics = [
        metric.model_copy(update={"reference": reference})
        for metric, reference in zip(record.metrics, references)
    ]
    return record.model_copy(update={"metrics": metrics})
