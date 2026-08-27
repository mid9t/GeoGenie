"""Vectorized NumPy brute-force SpatialIndex — competent baseline, not a strawman.

Beating a pure-Python loop is cheap; beating this is the honest speedup claim.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import POI
from geogenie.index._dist import sq_dist_to_points


class BruteForceIndex:
    """SpatialIndex: full scan via vectorized NumPy. Returns ids."""

    def __init__(self, origin: Optional[Origin] = None) -> None:
        self.origin: Optional[Origin] = origin
        self._xy: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._ids: np.ndarray = np.empty((0,), dtype=np.int64)
        self._by_id: Dict[int, POI] = {}
        self._pois: List[POI] = []

    def build(self, pois: list[POI]) -> None:
        if not pois:
            self._xy = np.empty((0, 2), dtype=np.float64)
            self._ids = np.empty((0,), dtype=np.int64)
            self._by_id = {}
            self._pois = []
            return
        if self.origin is None:
            lon0 = sum(p.lon for p in pois) / len(pois)
            lat0 = sum(p.lat for p in pois) / len(pois)
            self.origin = Origin(lon0, lat0)
        projected: List[POI] = []
        xs: List[float] = []
        ys: List[float] = []
        ids: List[int] = []
        for p in pois:
            x, y = to_xy(p.lon, p.lat, self.origin)
            projected.append(replace(p, x=x, y=y))
            xs.append(x)
            ys.append(y)
            ids.append(p.id)
        self._pois = projected
        self._by_id = {p.id: p for p in projected}
        self._xy = np.column_stack([xs, ys]).astype(np.float64)
        self._ids = np.asarray(ids, dtype=np.int64)

    def get_pois(self, ids: Sequence[int]) -> List[POI]:
        return [self._by_id[i] for i in ids if i in self._by_id]

    def range_query(self, x: float, y: float, radius_m: float) -> list[int]:
        if len(self._ids) == 0:
            return []
        r2 = radius_m * radius_m
        mask = sq_dist_to_points(x, y, self._xy) <= r2
        return self._ids[mask].tolist()

    def k_nearest(self, x: float, y: float, k: int) -> list[int]:
        if k <= 0 or len(self._ids) == 0:
            return []
        k = min(k, len(self._ids))
        d2 = sq_dist_to_points(x, y, self._xy)
        # argpartition then sort the k winners
        part = np.argpartition(d2, k - 1)[:k]
        order = part[np.argsort(d2[part])]
        return self._ids[order].tolist()
