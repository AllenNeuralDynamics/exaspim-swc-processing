"""Write each cell's reconstructions as Neuroglancer precomputed skeletons.

Neuroglancer cannot read SWC, so each coordinate space a cell is published in also gets a
standalone precomputed skeleton source, written with
`neuron-ng <https://github.com/AllenNeuralDynamics/neuron-ng>`_ beside the SWCs it is made
from::

    ccf_space_reconstructions/reconstructions.precomputed/
    specimen_space_reconstructions/reconstructions.precomputed/

Each source holds the cell's one reconstruction as segment :data:`SEGMENT_ID`, labelled with
its stem, and carries the SWC compartment code as a vertex attribute so a viewer can colour
by it. The CCF source is built from the CCF tracing and the specimen source from the
resampled specimen tracing. Both are in micrometers, so the source transform is a plain
micrometer-to-nanometer scale and the coordinates match the SWCs node for node.

Every asset holds one neuron, at the same relative paths, so quality control evaluated before
the asset is named can already point at a source by its asset-relative path; packaging then
makes the path absolute. See
:func:`~exaspim_swc_processing.quality_control.resolve_skeleton_references`.

Unlike regrouping files, converting them produces new data, so the conversion is recorded as
a :class:`~aind_data_schema.core.processing.DataProcess`; see :func:`conversion_process`.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess
from aind_data_schema_models.process_names import ProcessName
from neuron_ng import CoordinateSpace, Neuron, read_swc
from neuron_ng.precomputed import write_skeleton_source

from exaspim_swc_processing.stage import build_stage_process

logger = logging.getLogger(__name__)

PRECOMPUTED_DIRNAME = "reconstructions.precomputed"
"""Directory name of each precomputed skeleton source."""

SEGMENT_ID = 1
"""Segment id of the reconstruction in each source. An asset holds exactly one neuron."""

COMPARTMENT_ATTRIBUTE = "compartment"
"""Vertex attribute carrying the SWC compartment code, after ``radius``."""

SCALE_UM = (1.0, 1.0, 1.0)
"""Micrometers per SWC unit. Every reconstruction an asset publishes is in micrometers."""

CONVERSION_PROCESS_NAME = "precomputed_conversion"
"""Name of the process recording the conversion in each cell's ``processing.json``."""


@dataclass(frozen=True)
class PrecomputedSpec:
    """Describes one precomputed source a cell directory receives.

    Attributes
    ----------
    role : str
        Role of the artifact the source is built from, as in
        :data:`~exaspim_swc_processing.layout.ARTIFACT_SPECS`.
    destination : PurePosixPath
        Directory of the source, relative to the cell directory. This path is the contract
        with upstream quality control, which references the source by it.
    """

    role: str
    destination: PurePosixPath


PRECOMPUTED_SPECS: tuple[PrecomputedSpec, ...] = (
    PrecomputedSpec("ccf", PurePosixPath("ccf_space_reconstructions", PRECOMPUTED_DIRNAME)),
    PrecomputedSpec(
        "specimen_resampled",
        PurePosixPath("specimen_space_reconstructions", PRECOMPUTED_DIRNAME),
    ),
)
"""Sources written into each cell directory, one per coordinate space."""


def write_precomputed(swc: Path, out_dir: Path, label: str) -> None:
    """Convert one SWC into a precomputed skeleton source holding it as :data:`SEGMENT_ID`.

    The SWC is read before anything is written, so an invalid file (neuron-ng raises
    ``SWCError``, a ``ValueError``) leaves ``out_dir`` untouched.

    Parameters
    ----------
    swc : Path
        Reconstruction to convert, in micrometers.
    out_dir : Path
        Directory to write the source to. Anything already there is replaced, since the
        writer would otherwise leave stale segment files behind.
    label : str
        Label of the segment, shown by Neuroglancer.
    """
    skeleton = read_swc(swc)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    neuron = Neuron(
        segment_id=SEGMENT_ID,
        label=label,
        skeleton=skeleton,
        vertex_attributes={COMPARTMENT_ATTRIBUTE: skeleton.compartment.astype("<f4")},
    )
    write_skeleton_source(
        [neuron],
        out_dir,
        CoordinateSpace(SCALE_UM),
        attribute_names=[COMPARTMENT_ATTRIBUTE],
    )


def write_cell_precomputed(
    written: Mapping[str, Path],
    cell_dir: Path,
    label: str,
    specs: Iterable[PrecomputedSpec] = PRECOMPUTED_SPECS,
) -> dict[str, Path]:
    """Write a cell's precomputed sources from the reconstructions placed in its directory.

    A source whose reconstruction is missing or cannot be read is logged and skipped rather
    than failing the cell; the SWCs are still published.

    Parameters
    ----------
    written : Mapping[str, Path]
        Path of each reconstruction in the cell directory, keyed by role, as returned by
        :func:`~exaspim_swc_processing.layout.write_cell_directory`.
    cell_dir : Path
        The cell directory.
    label : str
        Segment label, typically the reconstruction stem.
    specs : Iterable[PrecomputedSpec], optional
        Sources to write, by default :data:`PRECOMPUTED_SPECS`.

    Returns
    -------
    dict[str, Path]
        Directory of each source written, keyed by role.
    """
    sources: dict[str, Path] = {}
    for spec in specs:
        swc = written.get(spec.role)
        if swc is None:
            logger.warning(
                "No %s reconstruction for %s; %s not written", spec.role, label, spec.destination
            )
            continue
        out_dir = cell_dir / spec.destination
        try:
            write_precomputed(swc, out_dir, label)
        except (OSError, ValueError) as error:
            logger.warning("Could not convert %s to precomputed: %s", swc, error)
            shutil.rmtree(out_dir, ignore_errors=True)
            continue
        sources[spec.role] = out_dir
    return sources


def conversion_process(code: Code, start_time: datetime) -> DataProcess:
    """Build the record of the SWC-to-precomputed conversion.

    Parameters
    ----------
    code : Code
        The code that ran the conversion, typically from
        :func:`~exaspim_swc_processing.stage.resolve_code`.
    start_time : datetime
        When the conversion started.

    Returns
    -------
    DataProcess
        A ``File format conversion`` process named :data:`CONVERSION_PROCESS_NAME`, to be
        appended to the stage records passed to
        :func:`~exaspim_swc_processing.packaging.package_cells`.
    """
    return build_stage_process(
        CONVERSION_PROCESS_NAME,
        code,
        start_time,
        process_type=ProcessName.FILE_FORMAT_CONVERSION,
    )
