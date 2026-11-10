"""Identifiers and asset names for exaSPIM SWC reconstructions.

Reconstruction files produced upstream are named ``<neuron-id>-<subject-id>-<annotator>.swc``,
for example ``N001-123456-HP.swc``. Every stage of the pipeline keys off that stem, so this
module turns it into a validated value object and derives the AIND asset names built from it.

The annotator identifies the upstream file only: packaged reconstructions are named
``<neuron-id>-<subject-id>``, e.g. ``N001-123456.swc``. See
:attr:`ReconstructionId.published_stem`.

AIND asset names use ``_`` to separate tokens, so no individual token may contain one. See
https://docs.allenneuraldynamics.org/en/latest/policies_practices/data_organization.html.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

from aind_data_schema_models.data_name_patterns import build_data_name

STEM_SEPARATOR = "-"
"""Separator between the tokens of a reconstruction file stem."""

STEM_TOKEN_COUNT = 3
"""Number of ``-``-separated tokens in a well-formed reconstruction file stem."""

PROCESS_LABEL_PREFIX = "reconstruction"
"""Prefix of the process label used when naming a per-cell derived asset."""

_SUBJECT_ID_PATTERN = re.compile(r"^\d+$")
_ILLEGAL_TOKEN_PATTERN = re.compile(r"[_/\\<>:;\"|?*]")


class ReconstructionNameError(ValueError):
    """Raised when a reconstruction file name does not follow the expected convention."""


@dataclass(frozen=True)
class ReconstructionId:
    """A parsed exaSPIM reconstruction identifier.

    Attributes
    ----------
    neuron_id : str
        Neuron identifier local to the subject, e.g. ``"N001"``.
    subject_id : str
        Numeric subject identifier, e.g. ``"123456"``. Kept as a string because it is an
        identifier rather than a quantity, and leading zeros are meaningful.
    annotator : str
        Initials of the annotator who traced the reconstruction, e.g. ``"HP"``.
    """

    neuron_id: str
    subject_id: str
    annotator: str

    @property
    def stem(self) -> str:
        """Reconstruction file stem this identifier was parsed from.

        Returns
        -------
        str
            The stem, e.g. ``"N001-123456-HP"``.
        """
        return STEM_SEPARATOR.join((self.neuron_id, self.subject_id, self.annotator))

    @property
    def published_stem(self) -> str:
        """Stem of the reconstruction files in the packaged asset.

        The annotator is dropped: it names the upstream file, not the neuron.

        Returns
        -------
        str
            The stem, e.g. ``"N001-123456"``.
        """
        return STEM_SEPARATOR.join((self.neuron_id, self.subject_id))

    @property
    def process_label(self) -> str:
        """Process label identifying this cell within a derived asset name.

        Uses ``-`` rather than ``_`` so the label stays a single AIND name token.

        Returns
        -------
        str
            The label, e.g. ``"reconstruction-N001"``.
        """
        return f"{PROCESS_LABEL_PREFIX}{STEM_SEPARATOR}{self.neuron_id}"


def _validate_token(token: str, field: str, stem: str) -> None:
    """Reject a stem token that is empty or contains a character illegal in AIND names.

    Parameters
    ----------
    token : str
        The token to validate.
    field : str
        Name of the field the token belongs to, used in the error message.
    stem : str
        The full stem, used in the error message.

    Raises
    ------
    ReconstructionNameError
        If the token is empty or contains an illegal character.
    """
    if not token:
        raise ReconstructionNameError(f"{field} is empty in reconstruction stem {stem!r}")
    illegal = _ILLEGAL_TOKEN_PATTERN.search(token)
    if illegal is not None:
        raise ReconstructionNameError(
            f"{field} {token!r} in reconstruction stem {stem!r} contains the illegal "
            f"character {illegal.group()!r}"
        )


def parse_stem(stem: str) -> ReconstructionId:
    """Parse a reconstruction file stem into its component identifiers.

    Parameters
    ----------
    stem : str
        Stem of a reconstruction file, without extension, e.g. ``"N001-123456-HP"``.

    Returns
    -------
    ReconstructionId
        The parsed identifier.

    Raises
    ------
    ReconstructionNameError
        If the stem does not have exactly three ``-``-separated tokens, if the subject
        token is not numeric, or if any token contains a character illegal in AIND names.

    Examples
    --------
    >>> parse_stem("N001-123456-HP")
    ReconstructionId(neuron_id='N001', subject_id='123456', annotator='HP')
    """
    tokens = stem.split(STEM_SEPARATOR)
    if len(tokens) != STEM_TOKEN_COUNT:
        raise ReconstructionNameError(
            f"Expected {STEM_TOKEN_COUNT} {STEM_SEPARATOR!r}-separated tokens "
            f"(<neuron>-<subject>-<annotator>) in reconstruction stem {stem!r}, "
            f"got {len(tokens)}"
        )
    neuron_id, subject_id, annotator = tokens
    _validate_token(neuron_id, "Neuron id", stem)
    _validate_token(subject_id, "Subject id", stem)
    _validate_token(annotator, "Annotator", stem)
    if _SUBJECT_ID_PATTERN.match(subject_id) is None:
        raise ReconstructionNameError(
            f"Subject id {subject_id!r} in reconstruction stem {stem!r} is not numeric"
        )
    return ReconstructionId(neuron_id=neuron_id, subject_id=subject_id, annotator=annotator)


def derived_asset_name(
    primary_asset_name: str,
    reconstruction: ReconstructionId,
    creation_time: datetime,
) -> str:
    """Build the AIND derived asset name for a single reconstruction.

    Follows ``<primary-asset-name>_<process-label>_<yyyy-mm-dd>_<HH-MM-SS>``, delegating the
    timestamp formatting to :func:`aind_data_schema_models.data_name_patterns.build_data_name`
    so it stays consistent with the rest of the AIND ecosystem.

    Parameters
    ----------
    primary_asset_name : str
        Name of the primary (raw) asset this reconstruction derives from, e.g.
        ``"exaSPIM_123456_2026-01-01_00-00-00"``.
    reconstruction : ReconstructionId
        Identifier of the reconstruction the asset describes.
    creation_time : datetime
        Time the derived asset was created. Use one value for every cell in a run so the
        whole run shares a timestamp.

    Returns
    -------
    str
        The derived asset name, e.g.
        ``"exaSPIM_123456_2026-01-01_00-00-00_reconstruction-N001_2026-02-01_00-00-00"``.

    Raises
    ------
    ReconstructionNameError
        If ``primary_asset_name`` is empty.
    """
    if not primary_asset_name:
        raise ReconstructionNameError("primary_asset_name is empty")
    label = f"{primary_asset_name}_{reconstruction.process_label}"
    return build_data_name(label, creation_time)


def asset_s3_uri(destination: str | None, asset_name: str) -> str | None:
    """Build the S3 URI an asset is uploaded to under ``destination``.

    Uses the same rule as the packaging capsule's upload: the asset directory is placed
    directly under the destination's prefix, which may be empty.

    Parameters
    ----------
    destination : str | None
        ``s3://bucket/prefix/`` the assets are uploaded under. Blank or ``None`` for a dry
        run, where nothing is uploaded.
    asset_name : str
        Name of the asset, which is also its directory name.

    Returns
    -------
    str | None
        ``s3://bucket/prefix/<asset_name>`` without a trailing slash, or ``None`` for a dry
        run.

    Raises
    ------
    ValueError
        If ``destination`` is not an ``s3://`` URI with a bucket.

    Examples
    --------
    >>> asset_s3_uri("s3://aind-open-data/", "exaSPIM_123456_x")
    's3://aind-open-data/exaSPIM_123456_x'
    """
    if not destination:
        return None
    parsed = urlparse(destination)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/prefix/ destination, got {destination!r}")
    key = "/".join(part for part in (parsed.path.strip("/"), asset_name) if part)
    return f"s3://{parsed.netloc}/{key}"
