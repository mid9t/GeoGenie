"""Spatial index Protocol — seam for k-d tree / R-tree / PostGIS."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from geogenie.core.types import POI


@runtime_checkable
class SpatialIndex(Protocol):
    def build(self, pois: list[POI]) -> None: ...

    def range_query(self, x: float, y: float, radius_m: float) -> list[POI]: ...

    def k_nearest(self, x: float, y: float, k: int) -> list[POI]: ...
