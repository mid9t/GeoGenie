"""Spatial index Protocol — seam for k-d tree / R-tree / PostGIS / brute force.

Query methods return POI ids (ints). Hydration is the store's job and must
not pollute index timings. [BENCHMARK_GAP Step 1]
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from geogenie.core.types import POI


@runtime_checkable
class SpatialIndex(Protocol):
    def build(self, pois: list[POI]) -> None: ...

    def range_query(self, x: float, y: float, radius_m: float) -> list[int]: ...

    def k_nearest(self, x: float, y: float, k: int) -> list[int]: ...
