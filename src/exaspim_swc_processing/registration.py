"""Recover the CCF registration's own parameters and reconstruct its reference geometry.
"""

from __future__ import annotations

import logging
import math
import re
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)

CCF_ALIGNMENT_DIR = "ccf_alignment"
"""Directory under a processed dataset holding the registration's outputs."""

PROCESSING_RECORD = f"{CCF_ALIGNMENT_DIR}/processing.json"
"""The registration's own record, relative to the dataset root."""

RESAMPLED_SPACING_MM = (0.01, 0.01, 0.01)
"""Isotropic spacing the registration resamples to, in millimetres."""

WRITTEN_SPACING_MM = {
    10: (0.010125, 0.010125, 0.0135),
    25: (0.02025, 0.02025, 0.027),
}
"""Spacing the registration writes into a loaded volume's header, by ``resolution_um``.

Fixed per pass in the registration's configuration, not read from the sample. The 10 um
entry matches all 48 published volumes; the 25 um entry is its configured counterpart and
is not exercised by the SWC transform.
"""

ANTS_DIRECTION = (
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)
"""Direction matrix as ANTs reports it, in LPS. Not the NIfTI RAS ``srow``."""

ORIGIN_MM = (0.0, 0.0, 0.0)
"""Origin of both reference volumes."""

_TEN_UM = 10
"""``resolution_um`` of the pass whose geometry the SWC transform consumes."""


class RegistrationRecordError(ValueError):
    """Raised when a registration record is missing or does not describe a usable pass."""


@dataclass(frozen=True)
class VolumeGeometry:
    """Geometry of one reference volume, without its voxels.

    Attributes
    ----------
    shape : tuple[int, int, int]
        Array shape in the order the registration wrote it, ``(x, z, y)``.
    spacing_mm : tuple[float, float, float]
        Voxel size in millimetres, matching ``shape``.
    """

    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]


@dataclass(frozen=True)
class RegistrationPass:
    """One registration pass, as the registration recorded it.

    Attributes
    ----------
    input_uri : str
        The zarr the registration read, e.g. ``s3://.../fusion/fused_ccf_ch.zarr/``.
    level : int
        Pyramid level within that zarr.
    resolution_um : int
        Target resolution of the pass, 10 or 25.
    sample_scale_mm : tuple[float, float, float]
        Voxel size as recorded, kept for provenance. Not used for geometry -- see
        :attr:`loaded_spacing_mm` for why it cannot be.
    sample_to_template : tuple[str, ...]
        Basenames of the sample-to-template transforms, resolved against
        ``<dataset>/ccf_alignment/``. The recorded paths point at Nextflow scratch.
    applied_spacing_mm : tuple[float, float, float] | None
        Spacing the registration set on the reoriented volume, in its axis order, or
        ``None`` for records written before the registration recorded it.
    """

    input_uri: str
    level: int
    resolution_um: int
    sample_scale_mm: tuple[float, float, float]
    sample_to_template: tuple[str, ...]
    applied_spacing_mm: tuple[float, float, float] | None = None

    @property
    def point_scale(self) -> tuple[float, float, float] | None:
        """Factors taking reoriented voxel indices to the pass's isotropic grid.

        This is ``applied_spacing_mm`` over the pass resolution, or ``None`` when the
        record predates ``applied_spacing_mm``.
        """
        if self.applied_spacing_mm is None:
            return None
        resolution_mm = self.resolution_um / 1000
        x, y, z = (spacing / resolution_mm for spacing in self.applied_spacing_mm)
        return (x, y, z)

    @property
    def loaded_spacing_mm(self) -> tuple[float, float, float]:
        """Spacing the registration wrote into the loaded volume's header.

        This is a constant of the pass, not a per-sample quantity: the registration
        configures a fixed ``sample_scale`` per resolution and writes it after applying
        the acquisition's axis swaps. All 48 published volumes carry
        ``(0.010125, 0.010125, 0.0135)``.

        It is deliberately not derived from the recorded ``sample_scale``, which is
        ambiguous. Some records store the raw configuration (coarse axis first) and
        others the reordered result, with nothing to distinguish them: 730904 and 826509
        have identical acquisition axes and therefore identical swaps, yet record
        ``(0.010125, 0.010125, 0.0135)`` and ``(0.0135, 0.010125, 0.010125)``
        respectively. Any rule reading that field is wrong for one of them.

        Note the written spacing does not correspond to the array's physical axes: the
        volume is stored ``(x, z, y)``, so the coarse value lands on axis 2 while the
        coarse physical axis is z at index 1. Reproducing this exactly is required, not
        optional -- the transforms were fit in this space, so a "corrected" spacing would
        invalidate them.

        Returns
        -------
        tuple[float, float, float]
            Spacing in millimetres, matching the written array axes.

        Raises
        ------
        RegistrationRecordError
            If no spacing is known for this pass's resolution.
        """
        try:
            return WRITTEN_SPACING_MM[self.resolution_um]
        except KeyError:
            raise RegistrationRecordError(
                f"No written spacing known for a {self.resolution_um} um pass; "
                f"known resolutions are {sorted(WRITTEN_SPACING_MM)}"
            ) from None


def _as_float_triple(values: object, field: str) -> tuple[float, float, float]:
    """Coerce a recorded three-element sequence to floats.

    Parameters
    ----------
    values : object
        The recorded value.
    field : str
        Field name, for the error message.

    Returns
    -------
    tuple[float, float, float]
        The three values.

    Raises
    ------
    RegistrationRecordError
        If the value is not a sequence of three numbers.
    """
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise RegistrationRecordError(f"{field} is not a three-element sequence: {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]))


def parse_registration_record(
    record: dict,
    resolution_um: int = _TEN_UM,
) -> RegistrationPass:
    """Extract one registration pass from ``ccf_alignment/processing.json``.

    The registration runs two passes, coarse then fine. The SWC transform consumes the
    fine one.

    Parameters
    ----------
    record : dict
        The decoded processing record.
    resolution_um : int, optional
        Which pass to take, by default 10.

    Returns
    -------
    RegistrationPass
        The recorded parameters of that pass.

    Raises
    ------
    RegistrationRecordError
        If no pass at that resolution is present, or it omits a required field.
    """
    for process in record.get("data_processes") or []:
        parameters = ((process.get("code") or {}).get("parameters")) or {}
        if parameters.get("resolution_um") != resolution_um:
            continue
        registration = parameters.get("sample_to_template_registration") or {}
        missing = [k for k in ("input_uri", "level") if parameters.get(k) is None]
        if missing:
            raise RegistrationRecordError(
                f"{resolution_um} um pass is missing {', '.join(missing)}"
            )
        return RegistrationPass(
            input_uri=str(parameters["input_uri"]),
            level=int(parameters["level"]),
            resolution_um=resolution_um,
            sample_scale_mm=_as_float_triple(registration.get("sample_scale"), "sample_scale"),
            sample_to_template=tuple(
                PurePosixPath(p).name for p in parameters.get("sample_to_template_transforms") or ()
            ),
            applied_spacing_mm=(
                _as_float_triple(registration["applied_spacing_mm"], "applied_spacing_mm")
                if registration.get("applied_spacing_mm") is not None
                else None
            ),
        )
    available = sorted(
        {
            ((p.get("code") or {}).get("parameters") or {}).get("resolution_um")
            for p in record.get("data_processes") or []
        }
        - {None}
    )
    raise RegistrationRecordError(
        f"No {resolution_um} um registration pass in the record; found {available or 'none'}"
    )


def loaded_geometry(pass_: RegistrationPass, zarr_shape: tuple[int, int, int]) -> VolumeGeometry:
    """Geometry of the volume the registration loaded.

    Parameters
    ----------
    pass_ : RegistrationPass
        The recorded pass.
    zarr_shape : tuple[int, int, int]
        Shape of ``<input_uri>/<level>``, in the zarr's ``(z, y, x)`` order.

    Returns
    -------
    VolumeGeometry
        Shape reordered to the written ``(x, z, y)``, with the recorded spacing.
    """
    z, y, x = zarr_shape
    return VolumeGeometry(shape=(x, z, y), spacing_mm=pass_.loaded_spacing_mm)


def resampled_geometry(loaded: VolumeGeometry) -> VolumeGeometry:
    """Geometry of the isotropically resampled volume.

    Shapes are computed with exact rationals and rounded half up. Binary floating point
    puts ``0.0135 / 0.01`` just below 1.35, which rounds a 1270-voxel axis down to 1714
    where the registration produced 1715.

    Parameters
    ----------
    loaded : VolumeGeometry
        Geometry of the loaded volume.

    Returns
    -------
    VolumeGeometry
        Shape scaled to isotropic spacing, with spacing :data:`RESAMPLED_SPACING_MM`.
    """
    factors = [
        Fraction(spacing).limit_denominator(10**6) / Fraction(target).limit_denominator(10**6)
        for spacing, target in zip(loaded.spacing_mm, RESAMPLED_SPACING_MM)
    ]
    shape = tuple(
        math.floor(Fraction(extent) * factor + Fraction(1, 2))
        for extent, factor in zip(loaded.shape, factors)
    )
    return VolumeGeometry(shape=shape, spacing_mm=RESAMPLED_SPACING_MM)


DATASET_NAME_PATTERN = re.compile(r"exaSPIM_\d+_[\d\-_]+_processed_[\d\-_]+")
"""Matches a processed dataset name wherever it appears in a path or URI."""


def dataset_name(*candidates: str) -> str | None:
    """Recover the processed dataset name from paths, URIs or a bare name.

    The transform may be pointed at an S3 URI, a dataset name, or a Code Ocean mount
    holding the same bundle. All three carry the name somewhere in the string.

    Parameters
    ----------
    *candidates : str
        Strings to search, in priority order.

    Returns
    -------
    str | None
        The first dataset name found, or ``None``.
    """
    for candidate in candidates:
        if not candidate:
            continue
        match = DATASET_NAME_PATTERN.search(candidate)
        if match:
            return match.group(0)
    return None


def zarr_level_key(pass_: RegistrationPass) -> str:
    """Key of the ``.zarray`` describing the level the registration read.

    Parameters
    ----------
    pass_ : RegistrationPass
        The recorded pass.

    Returns
    -------
    str
        Key relative to the bucket, e.g.
        ``exaSPIM_..._processed_.../fusion/fused_ccf_ch.zarr/2/.zarray``.
    """
    key = re.sub(r"^s3://[^/]+/", "", pass_.input_uri).rstrip("/")
    return f"{key}/{pass_.level}/.zarray"


NIFTI_HEADER_SIZE = 348
"""Bytes of a NIfTI-1 header, which carries everything needed here."""

HEADER_FETCH_BYTES = 200_000
"""Compressed bytes to request. Ample for the header of a gzipped NIfTI."""


def zarr_shape(zarray: dict) -> tuple[int, int, int]:
    """Read the spatial shape from a ``.zarray``, dropping leading singleton axes.

    Fused exaSPIM zarrs are written 5D as ``(t, c, z, y, x)``, so the spatial extent is
    the trailing three entries.

    Parameters
    ----------
    zarray : dict
        A decoded ``.zarray`` document.

    Returns
    -------
    tuple[int, int, int]
        Shape as ``(z, y, x)``, ready for :func:`loaded_geometry`.

    Raises
    ------
    RegistrationRecordError
        If the shape is missing, too short, or has a non-singleton leading axis.
    """
    shape = zarray.get("shape")
    if not isinstance(shape, list) or len(shape) < 3:
        raise RegistrationRecordError(f"Zarr shape {shape!r} does not describe a volume")
    leading, spatial = shape[:-3], shape[-3:]
    if any(extent != 1 for extent in leading):
        raise RegistrationRecordError(
            f"Zarr shape {shape} has a non-singleton leading axis; cannot reduce to 3D"
        )
    return (int(spatial[0]), int(spatial[1]), int(spatial[2]))


def parse_nifti_geometry(compressed_header: bytes) -> VolumeGeometry:
    """Read shape and spacing from the leading bytes of a gzipped NIfTI.

    Only the header is needed, so callers can fetch a byte range rather than a volume of
    well over a gigabyte.

    Parameters
    ----------
    compressed_header : bytes
        Leading bytes of the ``.nii.gz``. Need not be the whole file.

    Returns
    -------
    VolumeGeometry
        The volume's shape and spacing.

    Raises
    ------
    RegistrationRecordError
        If the bytes do not decompress to a readable NIfTI-1 header.
    """
    try:
        buffer = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(compressed_header, 4096)
    except zlib.error as error:
        raise RegistrationRecordError(f"Could not decompress the NIfTI header: {error}") from error
    if len(buffer) < NIFTI_HEADER_SIZE:
        raise RegistrationRecordError(
            f"Decompressed only {len(buffer)} bytes, need {NIFTI_HEADER_SIZE}"
        )
    endian = "<" if struct.unpack("<i", buffer[:4])[0] == NIFTI_HEADER_SIZE else ">"
    dims = struct.unpack(f"{endian}8h", buffer[40:56])
    pixdim = struct.unpack(f"{endian}8f", buffer[76:108])
    if dims[0] < 3:
        raise RegistrationRecordError(f"NIfTI header declares {dims[0]} dimensions, need 3")
    return VolumeGeometry(
        shape=(int(dims[1]), int(dims[2]), int(dims[3])),
        spacing_mm=(float(pixdim[1]), float(pixdim[2]), float(pixdim[3])),
    )


def registration_volume_key(dataset: str, dataset_id: str, volume: str) -> str:
    """Key of a published reference volume, if the registration wrote one.

    Parameters
    ----------
    dataset : str
        Processed dataset name.
    dataset_id : str
        Numeric subject id.
    volume : str
        Either ``"loaded"`` or ``"resampled"``.

    Returns
    -------
    str
        Key relative to the bucket.
    """
    return (
        f"{dataset}/{CCF_ALIGNMENT_DIR}/registration_metadata/"
        f"{dataset_id}_10um_{volume}_zarr_img.nii.gz"
    )


def reconcile(derived: VolumeGeometry, published: VolumeGeometry, volume: str) -> None:
    """Check a derived geometry against the one the registration published.

    Where both are available this turns the derivation from an assumption into a checked
    fact. A disagreement means the registration changed and the derivation is stale --
    which would otherwise yield plausible coordinates rather than an error.

    Parameters
    ----------
    derived : VolumeGeometry
        Geometry reconstructed from metadata.
    published : VolumeGeometry
        Geometry read from the published volume's header.
    volume : str
        Which volume, for the error message.

    Raises
    ------
    RegistrationRecordError
        If shape differs, or spacing differs by more than float32 precision.
    """
    if derived.shape != published.shape:
        raise RegistrationRecordError(
            f"Derived {volume} shape {derived.shape} does not match the published "
            f"{published.shape}; the registration's geometry has changed"
        )
    if any(abs(a - b) > 1e-6 for a, b in zip(derived.spacing_mm, published.spacing_mm)):
        raise RegistrationRecordError(
            f"Derived {volume} spacing {derived.spacing_mm} does not match the published "
            f"{published.spacing_mm}; the registration's geometry has changed"
        )


TEMPLATE_TO_CCF_PATTERN = re.compile(r"/data/(reg_exaspim_template_to_ccf_[A-Za-z0-9_.]+?)/")
"""A template-to-CCF data asset, as the registration names it in a mounted path."""


def recorded_template_to_ccf(record: dict) -> str:
    """Return the template-to-CCF asset the registration record says it used.

    The registration names the asset only by the path it mounted it at. Records in the v2
    schema list those paths in ``code.parameters.template_to_ccf_transforms``; records in
    the v1 schema mention them only in a process's ``notes``. Both are read, and every
    mention must name the same asset.

    The answer is only as good as the record. Registrations run between 2025-10-06 and
    the correction of their records mounted v1.5 at the ``..._v1.4`` path, so their
    records say v1.4.

    Parameters
    ----------
    record : dict
        The decoded ``ccf_alignment/processing.json``.

    Returns
    -------
    str
        The asset's mount name, e.g. ``"reg_exaspim_template_to_ccf_25um_v1.5"``.

    Raises
    ------
    RegistrationRecordError
        If the record names no template-to-CCF asset, or names more than one.
    """
    processes = record.get("data_processes") or (record.get("processing_pipeline") or {}).get(
        "data_processes", []
    )
    named: set[str] = set()
    for process in processes:
        parameters = (process.get("code") or {}).get("parameters") or process.get("parameters")
        for path in (parameters or {}).get("template_to_ccf_transforms") or []:
            named.update(TEMPLATE_TO_CCF_PATTERN.findall(f"{path}/"))
        named.update(TEMPLATE_TO_CCF_PATTERN.findall(str(process.get("notes") or "")))
    if not named:
        raise RegistrationRecordError("The registration record names no template-to-CCF asset")
    if len(named) > 1:
        raise RegistrationRecordError(
            f"The registration record names {len(named)} template-to-CCF assets: {sorted(named)}"
        )
    return named.pop()
