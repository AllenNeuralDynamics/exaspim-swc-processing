"""Tests for :mod:`exaspim_swc_processing.stage`."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from aind_data_schema.core.processing import DataProcess

from exaspim_swc_processing.stage import (
    DATA_PROCESS_FILENAME,
    UPSTREAM_STAGES,
    build_stage_process,
    carry_forward,
    installed_version,
    resolve_code,
    write_stage_process,
)

CAPSULE_ID = "7f4c2aad-f244-4c08-b443-80156da0ed4a"
COMMIT = "0d5ba138e7c9b7282d845e4dc6a8e2d213215716"
START = datetime(2026, 8, 19, 22, 10, 2, tzinfo=timezone.utc)


def test_code_url_names_the_capsule_that_actually_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old helper hardcoded slug 7989393 while running as capsule 4319239."""
    monkeypatch.setenv("CO_CAPSULE_ID", CAPSULE_ID)
    monkeypatch.delenv("CODE_VERSION", raising=False)
    assert resolve_code().url.endswith(CAPSULE_ID)


def test_code_version_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pinned commit is recorded, not whatever the remote HEAD is at run time."""
    monkeypatch.setenv("CO_CAPSULE_ID", CAPSULE_ID)
    monkeypatch.setenv("CODE_VERSION", COMMIT)
    assert resolve_code().version == COMMIT


def test_a_github_url_is_preferred_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """AIND prefers a repository URL over a Code Ocean one."""
    monkeypatch.setenv("CO_CAPSULE_ID", CAPSULE_ID)
    code = resolve_code(url="https://github.com/AllenNeuralDynamics/exaspim-swc-processing")
    assert code.url == "https://github.com/AllenNeuralDynamics/exaspim-swc-processing"


def test_version_falls_back_to_the_installed_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Code Ocean generates main.nf and does not export the pinned capsule commit.

    The installed library version is what actually determines a thin capsule's
    behaviour, and AIND accepts a version in place of a commit hash.
    """
    monkeypatch.setenv("CO_CAPSULE_ID", CAPSULE_ID)
    monkeypatch.delenv("CODE_VERSION", raising=False)
    assert resolve_code().version == installed_version("exaspim-swc-processing")


def test_version_is_none_when_nothing_identifies_the_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent version is null, not an empty string that looks recorded."""
    monkeypatch.setenv("CO_CAPSULE_ID", CAPSULE_ID)
    monkeypatch.delenv("CODE_VERSION", raising=False)
    assert resolve_code(distribution="not-installed-anywhere").version is None


def test_installed_version_returns_none_for_an_absent_distribution() -> None:
    """A capsule that did not install the library still produces a valid record."""
    assert installed_version("definitely-not-installed") is None


def test_url_is_empty_outside_code_ocean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running locally, with no capsule id and no url, leaves the url blank."""
    monkeypatch.delenv("CO_CAPSULE_ID", raising=False)
    assert resolve_code().url == ""


def test_a_stage_record_carries_only_the_required_fields() -> None:
    """Nothing optional is recorded unless the stage applied inputs."""
    process = build_stage_process("s", resolve_code(url="https://x"), START, ["Peter Grotz"])
    recorded = {k for k, v in process.model_dump().items() if v not in (None, [], {})}
    assert recorded == {
        "object_type",
        "process_type",
        "name",
        "stage",
        "code",
        "experimenters",
        "start_date_time",
    }
    assert {k for k, v in process.code.model_dump().items() if v is not None} == {
        "object_type",
        "url",
        "version",
    } - ({"version"} if process.code.version is None else set())


def test_applied_inputs_are_recorded_as_input_data() -> None:
    """The registration, template-to-CCF asset and any displacement field are recorded."""
    urls = ["s3://aind-open-data/d/ccf_alignment/", "reg_exaspim_template_to_ccf_25um_v1.5"]
    process = build_stage_process("s", resolve_code(url="https://x"), START, input_data=urls)
    assert [asset.url for asset in process.code.input_data] == urls


def test_named_parameters_are_recorded_even_when_empty() -> None:
    """A setting recorded as null says it was checked and absent, not forgotten."""
    process = build_stage_process(
        "s", resolve_code(url="https://x"), START, parameters={"displacement_field": None}
    )
    assert process.code.parameters.model_dump() == {"displacement_field": None}


def test_write_round_trips_through_the_schema(tmp_path: Path) -> None:
    """The written file is what the terminal stage will read back."""
    process = build_stage_process("s", resolve_code(url="https://x"), START)
    path = write_stage_process(process, tmp_path / "alignment")
    assert path.name == DATA_PROCESS_FILENAME
    reloaded = DataProcess.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded.name == "s"


def test_carry_forward_republishes_upstream_stages(tmp_path: Path) -> None:
    """Nextflow hands on only the previous stage's results, so each stage must republish."""
    data = tmp_path / "data"
    (data / "dispatch").mkdir(parents=True)
    (data / "dispatch" / "data_process.json").write_text("{}", encoding="utf-8")
    (data / "refinement" / "final-world").mkdir(parents=True)
    (data / "refinement" / "final-world" / "N001-794492-HP.swc").write_text("x", encoding="utf-8")

    results = tmp_path / "results"
    carried = carry_forward(data, results, UPSTREAM_STAGES)

    assert carried == ["dispatch", "refinement"]
    assert (results / "dispatch" / "data_process.json").is_file()
    assert (results / "refinement" / "final-world" / "N001-794492-HP.swc").is_file()


def test_carry_forward_skips_absent_stages(tmp_path: Path) -> None:
    """The first stage has no upstream outputs; that is not an error."""
    assert carry_forward(tmp_path / "data", tmp_path / "results", UPSTREAM_STAGES) == []


def test_carry_forward_preserves_nested_structure(tmp_path: Path) -> None:
    """Deeply nested outputs survive the hop."""
    data = tmp_path / "data"
    nested = data / "final" / "ccf_space_reconstructions" / "jsons"
    nested.mkdir(parents=True)
    (nested / "N001-794492-HP.json").write_text("{}", encoding="utf-8")
    results = tmp_path / "results"
    carry_forward(data, results, ("final",))
    assert (results / "final/ccf_space_reconstructions/jsons/N001-794492-HP.json").is_file()
