"""
2D k-d tree over planar (x, y) metres.

range_query / k_nearest return POI ids — hydration stays outside timed paths.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import POI
from geogenie.index._dist import sq_dist


@dataclass
class _Node:
    x: float
    y: float
    poi_id: int
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    axis: int = 0


class KDTreeIndex:
    """SpatialIndex: balanced 2D k-d tree in planar metres. Returns ids."""

    def __init__(self, origin: Optional[Origin] = None) -> None:
        self.origin: Optional[Origin] = origin
        self.root: Optional[_Node] = None
        self._pois: List[POI] = []
        self._by_id: Dict[int, POI] = {}

    def build(self, pois: list[POI]) -> None:
        if not pois:
            self.root = None
            self._pois = []
            self._by_id = {}
            return
        if self.origin is None:
            lon0 = sum(p.lon for p in pois) / len(pois)
            lat0 = sum(p.lat for p in pois) / len(pois)
            self.origin = Origin(lon0, lat0)
        projected: List[POI] = []
        for p in pois:
            x, y = to_xy(p.lon, p.lat, self.origin)
            projected.append(replace(p, x=x, y=y))
        self._pois = projected
        self._by_id = {p.id: p for p in projected}
        self.root = self._build(list(projected), depth=0)

    def get_pois(self, ids: Sequence[int]) -> List[POI]:
        return [self._by_id[i] for i in ids if i in self._by_id]

    def _build(self, pts: List[POI], depth: int) -> Optional[_Node]:
        if not pts:
            return None
        axis = depth % 2
        pts.sort(key=lambda p: (p.x if axis == 0 else p.y) or 0.0)
        mid = (len(pts) - 1) // 2
        p = pts[mid]
        node = _Node(x=p.x or 0.0, y=p.y or 0.0, poi_id=p.id, axis=axis)
        node.left = self._build(pts[:mid], depth + 1)
        node.right = self._build(pts[mid + 1 :], depth + 1)
        return node

    def range_query(self, x: float, y: float, radius_m: float) -> list[int]:
        r2 = radius_m * radius_m
        found: List[int] = []

        def _search(node: Optional[_Node]) -> None:
            if node is None:
                return
            if sq_dist(node.x, node.y, x, y) <= r2:
                found.append(node.poi_id)
            axis = node.axis
            diff = (x - node.x) if axis == 0 else (y - node.y)
            near = node.left if diff < 0 else node.right
            far = node.right if diff < 0 else node.left
            _search(near)
            if diff * diff <= r2:
                _search(far)

        _search(self.root)
        return found

    def k_nearest(self, x: float, y: float, k: int) -> list[int]:
        if k <= 0:
            return []
        heap: List[Tuple[float, int, int]] = []  # (-d, counter, id)
        counter = 0

        def _search(node: Optional[_Node]) -> None:
            nonlocal counter
            if node is None:
                return
            d = sq_dist(node.x, node.y, x, y)
            if len(heap) < k:
                heapq.heappush(heap, (-d, counter, node.poi_id))
                counter += 1
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, counter, node.poi_id))
                counter += 1
            axis = node.axis
            diff = (x - node.x) if axis == 0 else (y - node.y)
            near = node.left if diff < 0 else node.right
            far = node.right if diff < 0 else node.left
            _search(near)
            worst = -heap[0][0] if len(heap) == k else math.inf
            if diff * diff < worst:
                _search(far)

        _search(self.root)
        ordered = sorted(heap, key=lambda t: -t[0])
        return [poi_id for _, _, poi_id in ordered]


def brute_force_range(
    pois: Sequence[POI], x: float, y: float, radius_m: float
) -> List[POI]:
    """Linear-scan oracle referee (returns POIs). Kept for eval/oracle.py."""
    r2 = radius_m * radius_m
    out = []
    for p in pois:
        px, py = p.x or 0.0, p.y or 0.0
        if sq_dist(px, py, x, y) <= r2:
            out.append(p)
    return out
