"""Tests for :mod:`exaspim_swc_processing.naming`."""

import re
from datetime import datetime, timezone

import pytest
from aind_data_schema_models.data_name_patterns import DataRegex

from exaspim_swc_processing.naming import (
    ReconstructionId,
    ReconstructionNameError,
    asset_s3_uri,
    derived_asset_name,
    parse_stem,
)

PRIMARY_ASSET = "exaSPIM_794492_2026-01-09_16-50-40"
CREATION_TIME = datetime(2026, 8, 19, 22, 10, 2, tzinfo=timezone.utc)


def test_parse_stem_returns_components() -> None:
    """A well-formed stem parses into its three component identifiers."""
    assert parse_stem("N001-794492-HP") == ReconstructionId(
        neuron_id="N001", subject_id="794492", annotator="HP"
    )


def test_parse_stem_round_trips_through_stem_property() -> None:
    """Parsing a stem and reading it back returns the original string."""
    assert parse_stem("N067-794492-JT").stem == "N067-794492-JT"


def test_published_stem_drops_the_annotator() -> None:
    """Packaged files are named by neuron and subject only."""
    assert parse_stem("N024-841260-DR").published_stem == "N024-841260"


@pytest.mark.parametrize("stem", ["N001-794492", "N001", "N001-794492-HP-extra", ""])
def test_parse_stem_rejects_wrong_token_count(stem: str) -> None:
    """A stem without exactly three tokens is rejected."""
    with pytest.raises(ReconstructionNameError, match="tokens"):
        parse_stem(stem)


@pytest.mark.parametrize(
    ("stem", "field"),
    [("-794492-HP", "Neuron id"), ("N001--HP", "Subject id"), ("N001-794492-", "Annotator")],
)
def test_parse_stem_rejects_empty_token(stem: str, field: str) -> None:
    """An empty token is rejected and the error names the offending field."""
    with pytest.raises(ReconstructionNameError, match=f"{field} is empty"):
        parse_stem(stem)


def test_parse_stem_rejects_underscore_in_token() -> None:
    """Underscores separate AIND name tokens, so they may not appear inside one."""
    with pytest.raises(ReconstructionNameError, match="illegal character"):
        parse_stem("N_001-794492-HP")


def test_parse_stem_rejects_path_separator_in_token() -> None:
    """A character illegal in a filename is rejected."""
    with pytest.raises(ReconstructionNameError, match="illegal character"):
        parse_stem("N001-794492-H/P")


def test_parse_stem_rejects_non_numeric_subject_id() -> None:
    """The subject token must be numeric."""
    with pytest.raises(ReconstructionNameError, match="is not numeric"):
        parse_stem("N001-79A492-HP")


def test_process_label_uses_hyphen_to_stay_one_token() -> None:
    """The process label must not introduce an underscore into the asset name."""
    label = parse_stem("N001-794492-HP").process_label
    assert label == "reconstruction-N001"
    assert "_" not in label


def test_derived_asset_name_follows_aind_convention() -> None:
    """The derived name is ``<primary>_<process-label>_<date>_<time>``."""
    name = derived_asset_name(PRIMARY_ASSET, parse_stem("N001-794492-HP"), CREATION_TIME)
    expected = "exaSPIM_794492_2026-01-09_16-50-40_reconstruction-N001_2026-08-19_22-10-02"
    assert name == expected


def test_derived_asset_name_matches_aind_derived_regex() -> None:
    """The generated name round-trips through the schema's DERIVED pattern."""
    name = derived_asset_name(PRIMARY_ASSET, parse_stem("N001-794492-HP"), CREATION_TIME)
    match = re.match(DataRegex.DERIVED.value, name)
    assert match is not None
    assert match.group("input") == PRIMARY_ASSET
    assert match.group("process_name") == "reconstruction-N001"


def test_derived_asset_name_rejects_empty_primary_asset() -> None:
    """A derived name cannot be built without the primary asset it derives from."""
    with pytest.raises(ReconstructionNameError, match="primary_asset_name is empty"):
        derived_asset_name("", parse_stem("N001-794492-HP"), CREATION_TIME)


@pytest.mark.parametrize(
    ("destination", "expected"),
    [
        ("s3://aind-open-data/", "s3://aind-open-data/asset"),
        ("s3://aind-open-data", "s3://aind-open-data/asset"),
        ("s3://bucket/some/prefix/", "s3://bucket/some/prefix/asset"),
        ("s3://bucket//prefix", "s3://bucket/prefix/asset"),
    ],
)
def test_asset_s3_uri_places_the_asset_under_the_prefix(destination: str, expected: str) -> None:
    """The asset sits directly under the prefix, as the upload writes it.

    Parameters
    ----------
    destination : str
        Upload destination.
    expected : str
        URI of an asset named ``asset`` under it.
    """
    assert asset_s3_uri(destination, "asset") == expected


@pytest.mark.parametrize("destination", ["", None])
def test_asset_s3_uri_is_none_for_a_dry_run(destination: str | None) -> None:
    """No destination means nothing is uploaded, so there is no URI.

    Parameters
    ----------
    destination : str | None
        A blank destination.
    """
    assert asset_s3_uri(destination, "asset") is None


@pytest.mark.parametrize("destination", ["aind-open-data/", "https://bucket/prefix", "s3:///x"])
def test_asset_s3_uri_rejects_a_destination_that_is_not_s3(destination: str) -> None:
    """A malformed destination would publish links that point nowhere.

    Parameters
    ----------
    destination : str
        A destination that is not ``s3://bucket/...``.
    """
    with pytest.raises(ValueError, match="s3://bucket/prefix/"):
        asset_s3_uri(destination, "asset")
