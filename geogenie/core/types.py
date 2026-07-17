"""Shared frozen dataclasses for the GeoGenie backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class POI:
    id: int
    lon: float
    lat: float
    category: Optional[str] = None
    name: Optional[str] = None
    accessible: Optional[bool] = None
    noise_level: Optional[str] = None
    hours: Optional[str] = None
    # Planar metres relative to the index origin (filled at index build).
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass(frozen=True)
class StructuredQuery:
    origin_lon: float
    origin_lat: float
    radius_m: float
    minutes: Optional[float] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    sort_by: str = "distance"
    category: Optional[str] = None


@dataclass(frozen=True)
class SearchResult:
    poi: POI
    distance_m: float
    explanation: str = ""


@dataclass(frozen=True)
class ReachRing:
    """Reachability polygon as a closed ring of (lon, lat) vertices."""

    ring_lonlat: Tuple[Tuple[float, float], ...]
    origin_lon: float
    origin_lat: float
    minutes: float
    method: str = "alpha"

    def as_geojson(self) -> Dict[str, Any]:
        coords = [list(p) for p in self.ring_lonlat]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        return {
            "type": "Feature",
            "properties": {
                "origin": {"lon": self.origin_lon, "lat": self.origin_lat},
                "minutes": self.minutes,
                "method": self.method,
            },
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }


@dataclass(frozen=True)
class ReachResult:
    pois: Tuple[POI, ...]
    ring: ReachRing
    candidates: int = 0
    survivors: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
