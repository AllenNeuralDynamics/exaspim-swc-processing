"""Derive the AIND ``data_description`` for a single reconstruction.

Each cell directory is published as its own derived data asset, so each needs a
:class:`~aind_data_schema.core.data_description.DataDescription` derived from the imaging
asset the reconstructions came from.

Two details of the upstream data shape this module:

* exaSPIM ``_processed_`` assets declare ``data_level: raw`` even though they are derived
  from a raw acquisition. :func:`derive_data_description` dispatches on that field and
  would take the ``from_raw`` path, appending to the *whole* parent name and growing it by
  a segment per processing hop. We call
  :func:`~aind_data_schema.utils.inheritance.derive_data_description_from_derived`
  directly instead, which reparses the parent name and keeps derived names flat.
* That function requires ``data_level == DERIVED``, so the level is coerced on a **copy**.
  The caller's parent object is never mutated.
"""

from __future__ import annotations

from datetime import datetime

from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.utils.inheritance import derive_data_description_from_derived
from aind_data_schema_models.data_name_patterns import DataLevel

from exaspim_swc_processing.naming import ReconstructionId


class DataDescriptionDerivationError(ValueError):
    """Raised when a per-cell data description cannot be derived from its parent."""


def as_derived(parent: DataDescription) -> DataDescription:
    """Return a copy of ``parent`` whose ``data_level`` is ``DERIVED``.

    exaSPIM ``_processed_`` assets misdeclare themselves as ``raw``. Coercing on a copy
    lets us assert what the asset actually is without editing metadata we do not own.

    Parameters
    ----------
    parent : DataDescription
        The parent asset's data description.

    Returns
    -------
    DataDescription
        A copy with ``data_level`` set to ``DERIVED``. ``parent`` is unchanged.
    """
    if parent.data_level == DataLevel.DERIVED:
        return parent
    return parent.model_copy(update={"data_level": DataLevel.DERIVED})


def derive_cell_data_description(
    parent: DataDescription,
    reconstruction: ReconstructionId,
    creation_time: datetime,
    **overrides: object,
) -> DataDescription:
    """Derive the data description for one reconstruction's data asset.

    Parameters
    ----------
    parent : DataDescription
        Data description of the imaging asset the reconstructions derive from. Not mutated.
    reconstruction : ReconstructionId
        Identifier of the cell, supplying the process label.
    creation_time : datetime
        Creation time of the derived asset. Pass one value for every cell in a run so the
        whole run shares a timestamp.
    **overrides : object
        ``DataDescription`` fields to set explicitly, taking precedence over the values
        inherited from ``parent``. Use for fields the parent leaves empty, such as
        ``funding_source`` or ``modalities``.

    Returns
    -------
    DataDescription
        The derived description, named
        ``<raw-asset>_reconstruction-<neuron>_<yyyy-mm-dd>_<HH-MM-SS>``, with
        ``source_data`` recording the immediate parent rather than the raw asset.

    Raises
    ------
    DataDescriptionDerivationError
        If the parent name does not match the AIND derived-name pattern, so the original
        raw input cannot be recovered from it, or if a required field has no value in the
        parent and none was supplied in ``overrides``. A name in raw rather than derived
        form surfaces as a ``KeyError`` from ``parse_name``, which is wrapped here too.

    Examples
    --------
    >>> derive_cell_data_description(parent, parse_stem("N001-123456-HP"), when).name
    'exaSPIM_123456_2026-01-01_00-00-00_reconstruction-N001_2026-02-01_00-00-00'
    """
    try:
        return derive_data_description_from_derived(
            as_derived(parent),
            reconstruction.process_label,
            creation_time=creation_time,
            **overrides,
        )
    except (ValueError, KeyError) as error:
        raise DataDescriptionDerivationError(
            f"Could not derive a data description for {reconstruction.stem!r} from parent "
            f"{parent.name!r}: {error}"
        ) from error
