"""Tests for :mod:`exaspim_swc_processing.precomputed`."""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName

from exaspim_swc_processing.precomputed import (
    CONVERSION_PROCESS_NAME,
    PRECOMPUTED_SPECS,
    SEGMENT_ID,
    conversion_process,
    write_cell_precomputed,
    write_precomputed,
)

RESOURCES = Path(__file__).parent / "resources"
SWC = RESOURCES / "reconstruction.swc"
STEM = "N001-794492-HP"
SWC_XYZ = np.loadtxt(SWC, comments="#")[:, 2:5]


def _decode(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode a precomputed skeleton file with ``radius`` and one more attribute.

    Parameters
    ----------
    path : Path
        The encoded skeleton.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Vertex positions, edges, radii and compartment codes.
    """
    data = path.read_bytes()
    n_vertices, n_edges = np.frombuffer(data[:8], dtype="<u4")
    offset = 8
    xyz = np.frombuffer(data[offset : offset + 12 * n_vertices], dtype="<f4").reshape(-1, 3)
    offset += 12 * n_vertices
    edges = np.frombuffer(data[offset : offset + 8 * n_edges], dtype="<u4").reshape(-1, 2)
    offset += 8 * n_edges
    radius = np.frombuffer(data[offset : offset + 4 * n_vertices], dtype="<f4")
    offset += 4 * n_vertices
    compartment = np.frombuffer(data[offset : offset + 4 * n_vertices], dtype="<f4")
    assert offset + 4 * n_vertices == len(data)
    return xyz, edges, radius, compartment


def _written(cell_dir: Path, roles: tuple[str, ...] = ("ccf", "specimen_resampled")) -> dict:
    """Place the fixture SWC in a cell directory under each role, as packaging does.

    Parameters
    ----------
    cell_dir : Path
        The cell directory.
    roles : tuple[str, ...], optional
        Roles to place, by default one per precomputed source.

    Returns
    -------
    dict
        Path of each placed SWC, keyed by role.
    """
    written = {}
    for role in roles:
        path = cell_dir / role / f"{STEM}.swc"
        path.parent.mkdir(parents=True)
        shutil.copy(SWC, path)
        written[role] = path
    return written


def test_source_holds_the_reconstruction_as_one_segment(tmp_path: Path) -> None:
    """The source is in micrometers, labelled with the stem, and decodes to the SWC."""
    out = tmp_path / "reconstructions.precomputed"
    write_precomputed(SWC, out, STEM)

    info = json.loads((out / "info").read_text(encoding="utf-8"))
    assert info["@type"] == "neuroglancer_skeletons"
    assert info["transform"] == [1000.0, 0, 0, 0, 0, 1000.0, 0, 0, 0, 0, 1000.0, 0]
    assert [a["id"] for a in info["vertex_attributes"]] == ["radius", "compartment"]

    properties = json.loads((out / "segment_properties" / "info").read_text(encoding="utf-8"))
    assert properties["inline"]["ids"] == [str(SEGMENT_ID)]
    label = next(p for p in properties["inline"]["properties"] if p["id"] == "label")
    assert label["values"] == [STEM]

    xyz, edges, radius, compartment = _decode(out / str(SEGMENT_ID))
    np.testing.assert_allclose(xyz, SWC_XYZ, rtol=0, atol=1e-3)
    assert len(edges) == len(SWC_XYZ) - 1
    assert radius[0] == pytest.approx(5.0)
    assert compartment.tolist() == [1, 2, 2, 3, 3, 3]


def test_an_existing_source_is_replaced(tmp_path: Path) -> None:
    """Stale segment files from an earlier run do not survive."""
    out = tmp_path / "reconstructions.precomputed"
    out.mkdir()
    (out / "7").write_bytes(b"stale")
    write_precomputed(SWC, out, STEM)
    assert not (out / "7").exists()
    assert (out / str(SEGMENT_ID)).is_file()


def test_an_invalid_swc_writes_nothing(tmp_path: Path) -> None:
    """The SWC is read first, so a bad file leaves the destination as it was."""
    bad = tmp_path / "bad.swc"
    bad.write_text("not an swc", encoding="utf-8")
    out = tmp_path / "reconstructions.precomputed"
    out.mkdir()
    (out / "keep").write_bytes(b"")
    with pytest.raises(ValueError):
        write_precomputed(bad, out, STEM)
    assert (out / "keep").exists()


def test_each_space_gets_its_source(tmp_path: Path) -> None:
    """CCF and specimen sources are written beside their SWCs; the refined SWC is not used."""
    written = _written(tmp_path, ("ccf", "specimen_refined", "specimen_resampled"))
    sources = write_cell_precomputed(written, tmp_path, STEM)
    assert sources == {
        "ccf": tmp_path / "ccf_space_reconstructions" / "reconstructions.precomputed",
        "specimen_resampled": (
            tmp_path / "specimen_space_reconstructions" / "reconstructions.precomputed"
        ),
    }
    for source in sources.values():
        assert (source / "info").is_file()
        assert (source / str(SEGMENT_ID)).is_file()
    assert [str(spec.destination) for spec in PRECOMPUTED_SPECS] == [
        "ccf_space_reconstructions/reconstructions.precomputed",
        "specimen_space_reconstructions/reconstructions.precomputed",
    ]


def test_a_missing_reconstruction_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``specimen_resampled`` is optional upstream; its absence is logged, not fatal."""
    with caplog.at_level(logging.WARNING):
        sources = write_cell_precomputed(_written(tmp_path, ("ccf",)), tmp_path, STEM)
    assert set(sources) == {"ccf"}
    assert f"No specimen_resampled reconstruction for {STEM}" in caplog.text


def test_an_unreadable_reconstruction_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A bad SWC costs its source only; nothing half-written is left behind."""
    written = _written(tmp_path)
    written["ccf"].write_text("not an swc", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        sources = write_cell_precomputed(written, tmp_path, STEM)
    assert set(sources) == {"specimen_resampled"}
    assert not (tmp_path / "ccf_space_reconstructions").exists()
    assert "Could not convert" in caplog.text


def test_the_conversion_is_recorded_as_a_format_conversion() -> None:
    """The record says what was done, by which code, and when."""
    code = Code(url="https://github.com/AllenNeuralDynamics/exaspim-swc-processing", version="1")
    started = datetime(2026, 11, 10, tzinfo=timezone.utc)
    process = conversion_process(code, started)
    assert process.name == CONVERSION_PROCESS_NAME
    assert process.process_type == ProcessName.FILE_FORMAT_CONVERSION
    assert process.code.url == code.url
    assert process.start_date_time == started
