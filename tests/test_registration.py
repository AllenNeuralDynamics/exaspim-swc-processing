"""Tests for :mod:`exaspim_swc_processing.registration`."""

import gzip
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from exaspim_swc_processing.registration import (
    ANTS_DIRECTION,
    NIFTI_HEADER_SIZE,
    RegistrationRecordError,
    VolumeGeometry,
    dataset_name,
    loaded_geometry,
    parse_nifti_geometry,
    parse_registration_record,
    reconcile,
    recorded_template_to_ccf,
    registration_volume_key,
    resampled_geometry,
    zarr_level_key,
    zarr_shape,
)

FIXTURE = Path(__file__).parent / "resources" / "ccf_alignment_processing.json"
DATASET = "exaSPIM_826509_2026-05-12_18-03-50_processed_2026-05-22_14-04-22"


def _record() -> dict:
    """Load the real registration record for 826509.

    Returns
    -------
    dict
        The decoded ``ccf_alignment/processing.json``.
    """
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_ten_micron_pass_is_selected() -> None:
    """The registration runs coarse then fine; the SWC transform uses the fine pass."""
    result = parse_registration_record(_record())
    assert result.resolution_um == 10
    assert result.level == 2
    assert result.input_uri.endswith("fusion/fused_ccf_ch.zarr/")


def test_the_coarse_pass_can_be_selected_too() -> None:
    """The 25 um pass is recorded at a different level."""
    result = parse_registration_record(_record(), resolution_um=25)
    assert result.level == 3
    assert result.sample_scale_mm == (0.027, 0.02025, 0.02025)
    assert result.loaded_spacing_mm == (0.02025, 0.02025, 0.027)


def test_written_spacing_is_a_constant_of_the_pass() -> None:
    """All 48 published volumes carry this spacing, whatever the record says.

    This puts the coarse value on axis 2 although the coarse physical axis is z at
    index 1 -- an upstream quirk that must be reproduced, since the transforms were fit
    in this space.
    """
    result = parse_registration_record(_record())
    assert result.loaded_spacing_mm == (0.010125, 0.010125, 0.0135)


def test_written_spacing_ignores_the_recorded_sample_scale() -> None:
    """826509 records the raw config order; 730904 records the reordered result.

    Their acquisitions yield identical axis swaps, so no rule reading this field is
    right for both. The field is kept for provenance only.
    """
    result = parse_registration_record(_record())
    assert result.sample_scale_mm == (0.0135, 0.010125, 0.010125)
    assert result.loaded_spacing_mm != result.sample_scale_mm
    assert result.loaded_spacing_mm != tuple(reversed(result.sample_scale_mm))[::-1]


def test_a_pass_at_an_unknown_resolution_has_no_spacing() -> None:
    """Inventing a spacing would silently mis-register; fail instead."""
    pass_ = replace(parse_registration_record(_record()), resolution_um=17)
    with pytest.raises(RegistrationRecordError, match="No written spacing"):
        _ = pass_.loaded_spacing_mm


def test_transform_basenames_are_extracted() -> None:
    """Recorded paths point at Nextflow scratch, so only the basenames are usable."""
    result = parse_registration_record(_record())
    assert result.sample_to_template == (
        "826509_to_exaSPIM_SyN_1Warp.nii.gz",
        "826509_to_exaSPIM_SyN_0GenericAffine.mat",
    )


def test_an_absent_pass_is_reported_with_what_was_found() -> None:
    """A record with no matching pass names the resolutions it does have."""
    with pytest.raises(RegistrationRecordError, match="found"):
        parse_registration_record(_record(), resolution_um=99)


def test_a_pass_missing_required_fields_is_rejected() -> None:
    """A truncated record fails loudly rather than yielding partial geometry."""
    record = _record()
    for process in record["data_processes"]:
        params = (process.get("code") or {}).get("parameters") or {}
        if params.get("resolution_um") == 10:
            params.pop("level")
    with pytest.raises(RegistrationRecordError, match="level"):
        parse_registration_record(record)


def test_a_malformed_sample_scale_is_rejected() -> None:
    """A scale that is not three numbers is an error, not a silent default."""
    record = _record()
    for process in record["data_processes"]:
        params = (process.get("code") or {}).get("parameters") or {}
        if params.get("resolution_um") == 10:
            params["sample_to_template_registration"]["sample_scale"] = [0.01, 0.01]
    with pytest.raises(RegistrationRecordError, match="sample_scale"):
        parse_registration_record(record)


def test_zarr_level_key_points_at_the_level_the_registration_read() -> None:
    """The key is bucket-relative, so it can be fetched directly."""
    result = parse_registration_record(_record())
    assert zarr_level_key(result) == f"{DATASET}/fusion/fused_ccf_ch.zarr/2/.zarray"


def test_loaded_shape_is_the_zarr_shape_reordered() -> None:
    """Verified on 40 samples: the written volume is the zarr level as (x, z, y)."""
    result = parse_registration_record(_record())
    geometry = loaded_geometry(result, (646, 1267, 2061))
    assert geometry.shape == (2061, 646, 1267)
    assert geometry.spacing_mm == (0.010125, 0.010125, 0.0135)


def test_resampled_shape_uses_exact_rationals() -> None:
    """794492: (2061, 646, 1267) -> (2087, 654, 1710), verified against the real volume."""
    result = parse_registration_record(_record())
    loaded = loaded_geometry(result, (646, 1267, 2061))
    assert resampled_geometry(loaded).shape == (2087, 654, 1710)


def test_the_half_boundary_rounds_up_not_to_even() -> None:
    """1270 x 27/20 is exactly 1714.5.

    Python's ``round`` is banker's rounding and gives 1714; binary float division makes
    ``0.0135 / 0.01`` slightly less than 1.35, which also gives 1714. The registration
    produced 1715 for 791116 and 822177.
    """
    result = parse_registration_record(_record())
    loaded = loaded_geometry(result, (657, 1270, 2091))
    assert loaded.shape == (2091, 657, 1270)
    assert resampled_geometry(loaded).shape == (2117, 665, 1715)


def test_resampled_spacing_is_isotropic() -> None:
    """The resampled volume is 10 um on every axis."""
    result = parse_registration_record(_record())
    loaded = loaded_geometry(result, (646, 1267, 2061))
    assert resampled_geometry(loaded).spacing_mm == (0.01, 0.01, 0.01)


def test_direction_is_lps_not_the_nifti_srow() -> None:
    """ANTs negates the first two rows on read; using the srow mirrors x and y."""
    assert ANTS_DIRECTION == ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))


def _nifti_header(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    endian: str = "<",
) -> bytes:
    """Build a gzipped NIfTI-1 header carrying a given shape and spacing.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Array dimensions.
    spacing : tuple[float, float, float]
        Voxel size.
    endian : str, optional
        Byte order, by default little.

    Returns
    -------
    bytes
        The gzipped header.
    """
    buffer = bytearray(NIFTI_HEADER_SIZE)
    struct.pack_into(f"{endian}i", buffer, 0, NIFTI_HEADER_SIZE)
    struct.pack_into(f"{endian}8h", buffer, 40, 3, *shape, 1, 1, 1, 1)
    struct.pack_into(f"{endian}8f", buffer, 76, 1.0, *spacing, 0, 0, 0, 0)
    return gzip.compress(bytes(buffer))


def test_a_published_header_yields_its_geometry() -> None:
    """The header alone gives shape and spacing, so no volume need be fetched."""
    raw = _nifti_header((2061, 646, 1267), (0.010125, 0.010125, 0.0135))
    geometry = parse_nifti_geometry(raw)
    assert geometry.shape == (2061, 646, 1267)
    assert all(abs(a - b) < 1e-6 for a, b in zip(geometry.spacing_mm, (0.010125, 0.010125, 0.0135)))


def test_a_big_endian_header_is_read() -> None:
    """Byte order is taken from the header's own size field."""
    raw = _nifti_header((10, 20, 30), (1.0, 2.0, 3.0), endian=">")
    assert parse_nifti_geometry(raw).shape == (10, 20, 30)


def test_undecompressable_bytes_are_rejected() -> None:
    """A non-gzip response is an error, not a silent empty geometry."""
    with pytest.raises(RegistrationRecordError, match="decompress"):
        parse_nifti_geometry(b"not gzip at all")


def test_a_truncated_header_is_rejected() -> None:
    """Too few bytes must not be read as a partial header."""
    with pytest.raises(RegistrationRecordError, match="Decompressed only"):
        parse_nifti_geometry(gzip.compress(b"\x5c\x01\x00\x00" + b"\x00" * 40))


def test_a_header_with_too_few_dimensions_is_rejected() -> None:
    """A 2D volume cannot supply a 3D geometry."""
    buffer = bytearray(NIFTI_HEADER_SIZE)
    struct.pack_into("<i", buffer, 0, NIFTI_HEADER_SIZE)
    struct.pack_into("<8h", buffer, 40, 2, 10, 20, 1, 1, 1, 1, 1)
    with pytest.raises(RegistrationRecordError, match="dimensions"):
        parse_nifti_geometry(gzip.compress(bytes(buffer)))


def test_the_published_volume_key_is_built() -> None:
    """Older samples publish the volumes; newer ones do not."""
    assert registration_volume_key(DATASET, "826509", "loaded") == (
        f"{DATASET}/ccf_alignment/registration_metadata/826509_10um_loaded_zarr_img.nii.gz"
    )


def test_reconcile_accepts_a_matching_geometry() -> None:
    """Where both exist, the derivation is checked rather than assumed."""
    a = VolumeGeometry((2061, 646, 1267), (0.010125, 0.010125, 0.0135))
    b = VolumeGeometry((2061, 646, 1267), (0.010125000029802322, 0.010125, 0.0135))
    reconcile(a, b, "loaded")


def test_reconcile_rejects_a_changed_shape() -> None:
    """A shape difference means the registration changed; fail rather than mis-register."""
    a = VolumeGeometry((2061, 646, 1267), (0.010125, 0.010125, 0.0135))
    b = VolumeGeometry((2061, 646, 1268), (0.010125, 0.010125, 0.0135))
    with pytest.raises(RegistrationRecordError, match="shape"):
        reconcile(a, b, "loaded")


def test_reconcile_rejects_a_changed_spacing() -> None:
    """Spacing must agree to better than float32 precision."""
    a = VolumeGeometry((10, 10, 10), (0.01, 0.01, 0.01))
    b = VolumeGeometry((10, 10, 10), (0.01, 0.01, 0.02))
    with pytest.raises(RegistrationRecordError, match="spacing"):
        reconcile(a, b, "resampled")


def test_a_five_dimensional_zarr_shape_reduces_to_its_spatial_axes() -> None:
    """Fused exaSPIM zarrs are written ``(t, c, z, y, x)``; 826509 level 2."""
    assert zarr_shape({"shape": [1, 1, 663, 1213, 2005]}) == (663, 1213, 2005)


def test_a_three_dimensional_zarr_shape_passes_through() -> None:
    """A plain 3D zarr needs no reduction."""
    assert zarr_shape({"shape": [646, 1267, 2061]}) == (646, 1267, 2061)


def test_a_non_singleton_leading_axis_is_rejected() -> None:
    """A multi-channel array would make the spatial axes ambiguous."""
    with pytest.raises(RegistrationRecordError, match="non-singleton"):
        zarr_shape({"shape": [1, 3, 663, 1213, 2005]})


def test_a_shape_too_short_to_be_a_volume_is_rejected() -> None:
    """Two axes cannot give a 3D geometry."""
    with pytest.raises(RegistrationRecordError, match="does not describe a volume"):
        zarr_shape({"shape": [1213, 2005]})


def test_a_missing_zarr_shape_is_rejected() -> None:
    """An absent key is an error, not an empty shape."""
    with pytest.raises(RegistrationRecordError, match="does not describe a volume"):
        zarr_shape({})


def test_the_dataset_name_is_found_in_an_s3_uri() -> None:
    """The transform is usually pointed at the bundle, not the dataset root."""
    assert dataset_name(f"s3://aind-open-data/{DATASET}/ccf_alignment/") == DATASET


def test_the_dataset_name_is_found_in_a_mounted_path() -> None:
    """A Code Ocean mount carries the same name; S3 resolution cannot parse a path."""
    assert dataset_name(f"/data/{DATASET}/ccf_alignment") == DATASET


def test_a_bare_dataset_name_passes_through() -> None:
    """The name itself is already the answer."""
    assert dataset_name(DATASET) == DATASET


def test_candidates_are_tried_in_order() -> None:
    """Empty or unhelpful candidates fall through to the next."""
    assert dataset_name("", "/scratch/reg_bundle", f"/data/{DATASET}") == DATASET


def test_no_dataset_name_yields_none() -> None:
    """A caller must be able to tell that nothing was found."""
    assert dataset_name("/scratch/reg_bundle", "") is None


V1_FIXTURE = Path(__file__).parent / "resources" / "ccf_alignment_processing_v1.json"


def test_the_template_asset_is_read_from_a_v2_record() -> None:
    """826509's v2 record lists the asset in the pass parameters."""
    assert recorded_template_to_ccf(_record()) == "reg_exaspim_template_to_ccf_25um_v1.4"


def test_the_template_asset_is_read_from_a_v1_record() -> None:
    """782644's v1 record names the asset only in a process's notes."""
    record = json.loads(V1_FIXTURE.read_text(encoding="utf-8"))
    assert recorded_template_to_ccf(record) == "reg_exaspim_template_to_ccf_25um_v1.4"


def test_a_record_without_a_template_asset_is_refused() -> None:
    """721828's record names none; nothing is guessed."""
    record = {"processing_pipeline": {"data_processes": [{"name": "Image atlas alignment"}]}}
    with pytest.raises(RegistrationRecordError, match="names no"):
        recorded_template_to_ccf(record)


def test_a_record_naming_two_template_assets_is_refused() -> None:
    """A record that disagrees with itself cannot choose the transform."""
    record = _record()
    paths = record["data_processes"][0]["code"]["parameters"]["template_to_ccf_transforms"]
    paths[0] = "/data/reg_exaspim_template_to_ccf_25um_v1.5/1Warp.nii.gz"
    with pytest.raises(RegistrationRecordError, match="names 2"):
        recorded_template_to_ccf(record)
