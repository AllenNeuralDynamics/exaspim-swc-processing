"""Build stand-ins for the registration's reference volumes from their geometry alone.

See :mod:`exaspim_swc_processing.registration` for where the geometry comes from and how
it was verified.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np

from exaspim_swc_processing.registration import ANTS_DIRECTION, ORIGIN_MM, VolumeGeometry

FILL_VALUE = np.uint8(0)
"""Value every element of a stand-in array reports.

The dtype is the narrowest available. It is not meaningful -- no consumer reads an
element -- but a zero-strided view still reports an ``itemsize``, and code that multiplies
``size * itemsize`` to estimate memory should get a small number.
"""


class GeometryTarget(Protocol):
    """An image whose geometry can be set, as ``ants.ANTsImage`` can."""

    def set_spacing(self, spacing: tuple[float, float, float]) -> None:
        """Set the voxel size.

        Parameters
        ----------
        spacing : tuple[float, float, float]
            Voxel size in millimetres.
        """

    def set_origin(self, origin: tuple[float, float, float]) -> None:
        """Set the origin.

        Parameters
        ----------
        origin : tuple[float, float, float]
            Position of the first voxel, in millimetres.
        """

    def set_direction(self, direction: np.ndarray) -> None:
        """Set the direction matrix.

        Parameters
        ----------
        direction : np.ndarray
            A 3x3 matrix of direction cosines.
        """


TargetT = TypeVar("TargetT", bound=GeometryTarget)


def reference_array(geometry: VolumeGeometry) -> np.ndarray:
    """Return a read-only array with a geometry's shape and no backing storage.

    Parameters
    ----------
    geometry : VolumeGeometry
        The volume to imitate.

    Returns
    -------
    np.ndarray
        A constant, read-only view of the requested shape.
    """
    return np.broadcast_to(FILL_VALUE, geometry.shape)


def reference_arrays(
    loaded: VolumeGeometry, resampled: VolumeGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Return stand-ins for the loaded and resampled volumes.

    Parameters
    ----------
    loaded : VolumeGeometry
        Geometry of the volume the registration read from the zarr.
    resampled : VolumeGeometry
        Geometry of the isotropically resampled volume.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays shaped like the loaded and resampled volumes.
    """
    return reference_array(loaded), reference_array(resampled)


def apply_geometry(image: TargetT, geometry: VolumeGeometry) -> TargetT:
    """Stamp a volume's spacing, origin and direction onto an image.

    The image's own shape is irrelevant: the only consumer of this geometry converts
    indices to physical coordinates, which does not involve the shape.

    Parameters
    ----------
    image : TargetT
        An image accepting ``set_spacing``, ``set_origin`` and ``set_direction``.
    geometry : VolumeGeometry
        The geometry to apply.

    Returns
    -------
    TargetT
        The same image, modified in place and returned for convenience.
    """
    image.set_spacing(geometry.spacing_mm)
    image.set_origin(ORIGIN_MM)
    image.set_direction(np.array(ANTS_DIRECTION, dtype=float))
    return image
