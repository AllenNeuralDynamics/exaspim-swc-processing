"""Let a pipeline stage record what it did, as an AIND ``DataProcess``.

Each capsule writes a ``data_process.json`` beside its outputs; the terminal stage collects
them into each cell's :class:`~aind_data_schema.core.processing.Processing`. This replaces
the ``processing_metadata.py`` that was duplicated byte-for-byte across two capsules, whose
surrounding helpers had already drifted apart.

It also fixes how provenance was recorded. The old helper hardcoded a capsule slug — the
wrong one, as it happens: the transform capsule recorded slug ``7989393`` while running as
``4319239`` — and filled ``Code.version`` from ``git ls-remote HEAD`` at run time rather
than the commit the pipeline pinned. Both are read from the environment here, so a stage
reports the code that actually ran.

A record carries only what the schema requires, plus the inputs the stage applied
(``Code.input_data``) and any named settings it is asked to record (``Code.parameters``).
Lookup details belong in the run log, not the published record.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path

from aind_data_schema.components.identifiers import Code, DataAsset
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema_models.process_names import ProcessName

from exaspim_swc_processing.layout import place_artifact

CAPSULE_URL_TEMPLATE = "https://codeocean.allenneuraldynamics.org/capsule/{capsule_id}"
"""Code Ocean capsule URL, used when no repository URL is given."""

DATA_PROCESS_FILENAME = "data_process.json"
"""Name a stage writes its record under."""

UPSTREAM_STAGES = ("dispatch", "refinement", "alignment", "final")
"""Stage directories that are passed along the chain, in pipeline order."""

PROCESSED_DATASET_FILE = "processed_dataset.txt"
"""Hand-off between stages, not published metadata.

The transform writes the name of the processed dataset it applied here, in its
``alignment/`` output; packaging reads it to find the parent dataset.
"""


def installed_version(distribution: str) -> str | None:
    """Return the installed version of a distribution, if it is installed.

    Used as the fallback for ``Code.version``. Code Ocean generates ``main.nf`` and
    exports only ``CO_CAPSULE_ID``, ``CO_CPUS`` and ``CO_MEMORY``, so the pinned capsule
    commit is not available to the running code. The version of the library the capsule
    installed is available, is what actually determines behaviour for these thin
    capsules, and is a value AIND accepts in place of a commit hash.

    Parameters
    ----------
    distribution : str
        Distribution name, e.g. ``"exaspim-swc-processing"``.

    Returns
    -------
    str | None
        The installed version, or ``None`` if the distribution is not installed.
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def resolve_code(
    url: str | None = None,
    version: str | None = None,
    distribution: str = "exaspim-swc-processing",
) -> Code:
    """Describe the code a stage ran, from the environment.

    Parameters
    ----------
    url : str | None, optional
        Repository URL. AIND prefers a GitHub URL; when omitted the Code Ocean capsule
        URL is built from ``CO_CAPSULE_ID``.
    version : str | None, optional
        Version of the code that ran. When omitted, ``CODE_VERSION`` is read, then the
        installed version of ``distribution``. See :func:`installed_version`.
    distribution : str, optional
        Installed package to fall back to for the version, by default this library.

    Returns
    -------
    Code
        The URL (required) and version (which AIND will require) of the code that ran.
    """
    capsule_id = os.environ.get("CO_CAPSULE_ID", "")
    resolved_url = url or (CAPSULE_URL_TEMPLATE.format(capsule_id=capsule_id) if capsule_id else "")
    return Code(
        url=resolved_url,
        version=version or os.environ.get("CODE_VERSION") or installed_version(distribution),
    )


def build_stage_process(
    name: str,
    code: Code,
    start_time: datetime,
    experimenters: Sequence[str] = (),
    input_data: Sequence[str] = (),
    parameters: Mapping[str, object] | None = None,
    process_type: ProcessName = ProcessName.NEURON_SKELETON_PROCESSING,
) -> DataProcess:
    """Build the record describing one stage's work.

    Parameters
    ----------
    name : str
        Name of the stage. Must be unique within a run, since
        :class:`~aind_data_schema.core.processing.Processing` rejects duplicates.
    code : Code
        The code that ran, from :func:`resolve_code`.
    start_time : datetime
        When the stage started.
    experimenters : Sequence[str], optional
        Who is responsible for the run.
    input_data : Sequence[str], optional
        URLs of the data the stage applied, recorded as ``Code.input_data``.
    parameters : Mapping[str, object] | None, optional
        Named settings worth recording even when empty, as ``Code.parameters``.
    process_type : ProcessName, optional
        Operation performed, by default ``NEURON_SKELETON_PROCESSING``.

    Returns
    -------
    DataProcess
        The stage record, ready to write with :func:`write_stage_process`.
    """
    extra: dict[str, object] = {}
    if input_data:
        extra["input_data"] = [DataAsset(url=url) for url in input_data]
    if parameters:
        extra["parameters"] = dict(parameters)
    if extra:
        # Revalidated: model_copy would leave plain dicts where the schema expects models.
        code = Code.model_validate({**code.model_dump(), **extra})
    return DataProcess(
        process_type=process_type,
        name=name,
        stage=ProcessStage.PROCESSING,
        code=code,
        experimenters=list(experimenters),
        start_date_time=start_time,
    )


def write_stage_process(process: DataProcess, output_dir: Path) -> Path:
    """Write a stage record beside the stage's outputs.

    Parameters
    ----------
    process : DataProcess
        The record to write.
    output_dir : Path
        Directory to write into. Created if absent.

    Returns
    -------
    Path
        The file written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / DATA_PROCESS_FILENAME
    path.write_text(process.model_dump_json(indent=2), encoding="utf-8")
    return path


def carry_forward(
    data_dir: Path,
    results_dir: Path,
    stages: Sequence[str],
) -> list[str]:
    """Republish upstream stage outputs so later stages can still see them.

    Nextflow hands each process only the previous one's ``/results``, so a stage that does
    not republish what it received removes it from the chain. The terminal packaging stage
    needs every stage's ``data_process.json`` and the refined reconstructions, so each
    intermediate stage has to pass them along.

    Files are hardlinked where the filesystem allows and copied otherwise, so passing a
    multi-gigabyte tree through costs little beyond directory entries.

    Parameters
    ----------
    data_dir : Path
        Directory the upstream outputs are mounted at.
    results_dir : Path
        Directory this stage writes to.
    stages : Sequence[str]
        Stage directory names to republish, if present.

    Returns
    -------
    list[str]
        The stages that were found and republished, in the order given.
    """
    carried: list[str] = []
    for stage in stages:
        source = data_dir / stage
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_file():
                place_artifact(path, results_dir / stage / path.relative_to(source))
        carried.append(stage)
    return carried
