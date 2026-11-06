"""Resolve the parent asset's ``data_description``, recording where it came from.

Per-cell assets derive from the imaging asset the reconstructions were traced on, so its
``data_description`` has to be found before anything can be written. Three sources are
tried in order, because exaSPIM coverage is uneven:

1. **DocDB v2** — already migrated centrally. Preferred: nothing local to do.
2. **DocDB v1** — the original document, upgraded locally with ``aind-metadata-upgrader``.
3. **S3** — the bucket the indexer itself reads, for assets not yet registered.

Which source answered is recorded in the output, so assets built from a fallback can be
found and re-derived once the upstream registration catches up. See
:meth:`ParentMetadata.provenance` and :meth:`ParentMetadata.tags`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from aind_data_schema.core.data_description import DataDescription

logger = logging.getLogger(__name__)

TAG_PREFIX = "metadata-source"
"""Prefix of the ``data_description`` tag naming the source."""

UPGRADED_TAG = "metadata-upgraded"
"""Tag applied when the parent document was upgraded locally rather than read as v2."""


class MetadataSource(str, Enum):
    """Where a parent ``data_description`` was read from."""

    DOCDB_V2 = "docdb_v2"
    DOCDB_V1 = "docdb_v1"
    S3 = "s3"


class ParentMetadataNotFoundError(LookupError):
    """Raised when no source holds a ``data_description`` for the parent asset."""


@dataclass(frozen=True)
class ParentMetadata:
    """A parent ``data_description`` together with where it came from.

    Attributes
    ----------
    data_description : DataDescription
        The parent's description, upgraded to v2 if it was not already.
    asset_name : str
        Name of the parent asset it describes.
    source : MetadataSource
        Which source answered.
    source_schema_version : str | None
        ``schema_version`` of the document as read, before any upgrade.
    upgraded : bool
        Whether the document was upgraded locally.
    """

    data_description: DataDescription
    asset_name: str
    source: MetadataSource
    source_schema_version: str | None
    upgraded: bool

    def provenance(self) -> dict[str, object]:
        """Render the source as fields for a ``DataProcess.output_parameters``.

        Returns
        -------
        dict[str, object]
            Keys ``parent_asset_name``, ``metadata_source``, ``parent_schema_version``
            and ``upgraded_to``. ``upgraded_to`` is ``None`` when nothing was upgraded.
        """
        return {
            "parent_asset_name": self.asset_name,
            "metadata_source": self.source.value,
            "parent_schema_version": self.source_schema_version,
            "upgraded_to": self.data_description.schema_version if self.upgraded else None,
        }

    def tags(self) -> list[str]:
        """Render the source as ``data_description`` tags.

        Tags are indexed, so these are what make assets built from a fallback findable
        once the upstream registration catches up.

        Returns
        -------
        list[str]
            ``["metadata-source:<source>"]``, plus ``"metadata-upgraded"`` if upgraded.
        """
        tags = [f"{TAG_PREFIX}:{self.source.value}"]
        if self.upgraded:
            tags.append(UPGRADED_TAG)
        return tags


Fetcher = Callable[[str], dict | None]
"""Returns a raw ``data_description`` document for an asset name, or ``None``."""

Upgrader = Callable[[dict], dict]
"""Upgrades a raw ``data_description`` document to the current schema."""


MINIMUM_CURRENT_MAJOR = 2
"""Major ``schema_version`` at or above which a document needs no upgrade."""


def _is_current_schema(document: dict) -> bool:
    """Report whether a raw document is new enough to use without upgrading.

    Compares the major version rather than matching ``"2."``, so a document written
    against a future schema is left alone rather than pushed through an upgrader that
    only knows how to move forwards.

    Parameters
    ----------
    document : dict
        Raw ``data_description`` payload.

    Returns
    -------
    bool
        ``True`` if the major ``schema_version`` is at least
        :data:`MINIMUM_CURRENT_MAJOR`. A missing or unparseable version is treated as old.
    """
    major, _, _ = str(document.get("schema_version", "")).partition(".")
    if not major.isdigit():
        return False
    return int(major) >= MINIMUM_CURRENT_MAJOR


def _with_asset_name(payload: dict, asset_name: str) -> dict:
    """Return ``payload`` with its ``name`` set to the asset it was fetched for.

    exaSPIM ``_processed_`` assets carry the *raw* asset's name in their
    ``data_description``, so the document disagrees with the asset it describes. Deriving
    a child from the raw-form name fails, because the original input cannot be parsed out
    of it. The name we asked for is the authoritative one; ``aind-metadata-upgrader``
    resolves the same conflict the same way.

    Parameters
    ----------
    payload : dict
        Raw ``data_description`` payload.
    asset_name : str
        Name of the asset it was fetched for.

    Returns
    -------
    dict
        The payload, with ``name`` corrected if it differed.
    """
    if payload.get("name") == asset_name:
        return payload
    logger.info(
        "Parent document names itself %r but was fetched as %r; using the latter",
        payload.get("name"),
        asset_name,
    )
    return {**payload, "name": asset_name}


def resolve_parent_metadata(
    asset_name: str,
    sources: Iterable[tuple[MetadataSource, Fetcher]],
    upgrader: Upgrader,
) -> ParentMetadata:
    """Find the parent's ``data_description``, trying each source in order.

    Parameters
    ----------
    asset_name : str
        Name of the parent asset.
    sources : Iterable[tuple[MetadataSource, Fetcher]]
        Sources to try, in priority order. Each fetcher returns a raw document or ``None``.
    upgrader : Upgrader
        Applied to documents that are not already v2.

    Returns
    -------
    ParentMetadata
        The description and the source that answered.

    Raises
    ------
    ParentMetadataNotFoundError
        If no source returned a document. The caller should skip the sample rather than
        publish a derived asset with no provenance.
    """
    attempted: list[str] = []
    for source, fetch in sources:
        attempted.append(source.value)
        document = fetch(asset_name)
        if document is None:
            continue
        original_version = document.get("schema_version")
        upgraded = not _is_current_schema(document)
        payload = upgrader(document) if upgraded else document
        payload = _with_asset_name(payload, asset_name)
        return ParentMetadata(
            data_description=DataDescription.model_validate(payload),
            asset_name=asset_name,
            source=source,
            source_schema_version=(str(original_version) if original_version is not None else None),
            upgraded=upgraded,
        )
    raise ParentMetadataNotFoundError(
        f"No data_description found for parent asset {asset_name!r}; tried " + ", ".join(attempted)
    )
