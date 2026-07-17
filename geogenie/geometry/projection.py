"""
Local tangent-plane projection: lat/lon <-> metres.

Why this module exists
----------------------
Every classical geometry algorithm (convex hull, Douglas-Peucker, Delaunay,
Voronoi) assumes a *Euclidean* plane where the x and y axes share a unit.
Latitude/longitude does not satisfy that: one degree of latitude is ~111 km
everywhere, but one degree of longitude is 111*cos(lat) km. At 50 deg N a
degree of longitude is only ~64% of a degree of latitude.

Run a convex hull on raw degrees and it will not crash -- it will return a
hull that is subtly wrong, sheared along the x axis. Run Douglas-Peucker on
raw degrees and "tolerance" has no physical meaning. Run Voronoi on raw
degrees and the cell boundaries are not equidistant in the real world.

So: project once, compute in metres, unproject at the boundary.

Choice of projection
--------------------
Local equirectangular (a.k.a. "flat-Earth" / plate carree about an origin):

    x = R * (lon - lon0) * cos(lat0)
    y = R * (lat - lat0)

This is a first-order Taylor expansion of the sphere about (lat0, lon0). It
is cheap, exactly invertible, and has no singularities away from the poles.

Accuracy: distance error grows roughly as (d/R)^2. Empirically (see
tests/test_projection.py) it holds well under ~0.1% relative error out to
~25 km from the origin, which comfortably covers every use case GeoGenie
has -- a 10-minute walk is ~800 m, and even a city-scale bounding box is
tens of km. If this project ever needs country-scale geometry, swap this
class for a proper transverse Mercator / UTM implementation; the rest of
the geometry package is unit-agnostic and will not need to change.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

# IUGG mean Earth radius, metres.
R_EARTH = 6_371_008.8

__all__ = ["R_EARTH", "LocalTangentPlane", "haversine"]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres. Reference implementation, used to
    validate the projection -- not on any hot path."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


class LocalTangentPlane:
    """Bidirectional lat/lon <-> metric (x, y) about a fixed origin.

    x is eastward metres, y is northward metres, origin maps to (0, 0).

    The origin must be held fixed for the lifetime of any set of coordinates
    you intend to compare. Two points projected through different origins are
    not in the same coordinate system and must never be mixed. This class is
    immutable to make that discipline easy to keep.
    """

    __slots__ = ("lat0", "lon0", "_kx", "_ky")

    def __init__(self, lat0: float, lon0: float) -> None:
        if not -90.0 <= lat0 <= 90.0:
            raise ValueError(f"lat0 out of range: {lat0}")
        if not -180.0 <= lon0 <= 180.0:
            raise ValueError(f"lon0 out of range: {lon0}")
        if abs(lat0) > 85.0:
            # cos(lat0) -> 0; longitude scale collapses and the inverse
            # becomes numerically unstable. Fail loudly rather than return
            # garbage.
            raise ValueError(
                f"LocalTangentPlane is unusable near the poles (lat0={lat0}). "
                "Use a polar stereographic projection instead."
            )
        self.lat0 = lat0
        self.lon0 = lon0
        # Metres per degree, precomputed.
        self._ky = R_EARTH * math.pi / 180.0
        self._kx = self._ky * math.cos(math.radians(lat0))

    # -- construction -----------------------------------------------------

    @classmethod
    def from_points(cls, latlons: Iterable[Tuple[float, float]]) -> "LocalTangentPlane":
        """Origin at the centroid of the bounding box of the given points.

        Centring the origin halves the maximum distance from it, which
        quarters the worst-case projection error.
        """
        lats: List[float] = []
        lons: List[float] = []
        for lat, lon in latlons:
            lats.append(lat)
            lons.append(lon)
        if not lats:
            raise ValueError("from_points() requires at least one point")
        return cls((min(lats) + max(lats)) / 2.0, (min(lons) + max(lons)) / 2.0)

    # -- forward ----------------------------------------------------------

    def to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        dlon = _wrap_lon(lon - self.lon0)
        return (dlon * self._kx, (lat - self.lat0) * self._ky)

    def to_xy_many(self, latlons: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        kx, ky, lat0, lon0 = self._kx, self._ky, self.lat0, self.lon0
        return [(_wrap_lon(lon - lon0) * kx, (lat - lat0) * ky) for lat, lon in latlons]

    # -- inverse ----------------------------------------------------------

    def to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        return (y / self._ky + self.lat0, _wrap_lon(x / self._kx + self.lon0))

    def to_latlon_many(self, xys: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        kx, ky, lat0, lon0 = self._kx, self._ky, self.lat0, self.lon0
        return [(y / ky + lat0, _wrap_lon(x / kx + lon0)) for x, y in xys]

    def __repr__(self) -> str:  # pragma: no cover
        return f"LocalTangentPlane(lat0={self.lat0:.6f}, lon0={self.lon0:.6f})"


def _wrap_lon(dlon: float) -> float:
    """Normalise a longitude delta to [-180, 180) so the antimeridian does
    not produce a 40 000 km jump."""
    return (dlon + 180.0) % 360.0 - 180.0
