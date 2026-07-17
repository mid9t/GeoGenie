"""
Coordinate funnel — the ONLY place lon/lat becomes planar metres.

Every other GeoGenie module must import to_xy / to_lonlat from here.
Direct use of geometry.projection outside this file is a CI failure.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

from geogenie.geometry.projection import LocalTangentPlane, haversine  # noqa: F401

# Re-export haversine for oracles; callers still go through this module.
__all__ = ["Origin", "to_xy", "to_lonlat", "haversine", "MAX_SAFE_LAT"]

# Stricter than geometry.LocalTangentPlane's 85° guard. [VR §3.1]
MAX_SAFE_LAT = 70.0


class Origin(NamedTuple):
    lon: float
    lat: float


def _validate_lat(lat: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude out of range [-90, 90]: {lat}")
    if abs(lat) > MAX_SAFE_LAT:
        raise ValueError(
            f"latitude |{lat}| exceeds local-tangent safe limit "
            f"({MAX_SAFE_LAT}°). Use a proper map projection."
        )


def _validate_lon(lon: float) -> None:
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude out of range [-180, 180]: {lon}")


def to_xy(lon: float, lat: float, origin: Origin) -> Tuple[float, float]:
    """Project (lon, lat) degrees to planar (x, y) metres about origin.

    Argument order is always (lon, lat). The geometry package's
    LocalTangentPlane.to_xy takes (lat, lon) — that swap happens here only.
    """
    _validate_lon(lon)
    _validate_lat(lat)
    _validate_lon(origin.lon)
    _validate_lat(origin.lat)
    plane = LocalTangentPlane(origin.lat, origin.lon)
    return plane.to_xy(lat, lon)


def to_lonlat(x: float, y: float, origin: Origin) -> Tuple[float, float]:
    """Inverse of to_xy. Returns (lon, lat)."""
    _validate_lon(origin.lon)
    _validate_lat(origin.lat)
    plane = LocalTangentPlane(origin.lat, origin.lon)
    lat, lon = plane.to_latlon(x, y)
    return (lon, lat)
