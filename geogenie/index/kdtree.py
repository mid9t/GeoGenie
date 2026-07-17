"""
2D k-d tree over planar (x, y) metres, adapted from Phase 1 spatial_index.

Stores POI objects; range_query is a circle in planar metres.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import POI


@dataclass
class _Node:
    poi: POI
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    axis: int = 0


class KDTreeIndex:
    """SpatialIndex implementation: balanced 2D k-d tree in planar metres."""

    def __init__(self, origin: Optional[Origin] = None) -> None:
        self.origin: Optional[Origin] = origin
        self.root: Optional[_Node] = None
        self._pois: List[POI] = []

    def build(self, pois: list[POI]) -> None:
        if not pois:
            self.root = None
            self._pois = []
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
        self.root = self._build(list(projected), depth=0)

    def _build(self, pts: List[POI], depth: int) -> Optional[_Node]:
        if not pts:
            return None
        axis = depth % 2
        pts.sort(key=lambda p: (p.x if axis == 0 else p.y) or 0.0)
        mid = (len(pts) - 1) // 2
        node = _Node(poi=pts[mid], axis=axis)
        node.left = self._build(pts[:mid], depth + 1)
        node.right = self._build(pts[mid + 1 :], depth + 1)
        return node

    def range_query(self, x: float, y: float, radius_m: float) -> list[POI]:
        """Circle query in planar metres."""
        r2 = radius_m * radius_m
        found: List[POI] = []

        def _search(node: Optional[_Node]) -> None:
            if node is None:
                return
            px, py = node.poi.x or 0.0, node.poi.y or 0.0
            dx, dy = px - x, py - y
            if dx * dx + dy * dy <= r2:
                found.append(node.poi)
            axis = node.axis
            diff = (x - px) if axis == 0 else (y - py)
            near = node.left if diff < 0 else node.right
            far = node.right if diff < 0 else node.left
            _search(near)
            # Far branch can intersect the circle if split plane is within radius.
            if diff * diff <= r2:
                _search(far)

        _search(self.root)
        return found

    def k_nearest(self, x: float, y: float, k: int) -> list[POI]:
        if k <= 0:
            return []
        heap: List[Tuple[float, int, POI]] = []  # max-heap via neg dist
        counter = 0

        def _search(node: Optional[_Node]) -> None:
            nonlocal counter
            if node is None:
                return
            px, py = node.poi.x or 0.0, node.poi.y or 0.0
            d = (px - x) ** 2 + (py - y) ** 2
            if len(heap) < k:
                heapq.heappush(heap, (-d, counter, node.poi))
                counter += 1
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, counter, node.poi))
                counter += 1
            axis = node.axis
            diff = (x - px) if axis == 0 else (y - py)
            near = node.left if diff < 0 else node.right
            far = node.right if diff < 0 else node.left
            _search(near)
            worst = -heap[0][0] if len(heap) == k else math.inf
            if diff * diff < worst:
                _search(far)

        _search(self.root)
        ordered = sorted(heap, key=lambda t: -t[0])
        return [poi for _, _, poi in ordered]


def brute_force_range(
    pois: Sequence[POI], x: float, y: float, radius_m: float
) -> List[POI]:
    """Linear-scan oracle for range_query."""
    r2 = radius_m * radius_m
    out = []
    for p in pois:
        px, py = p.x or 0.0, p.y or 0.0
        if (px - x) ** 2 + (py - y) ** 2 <= r2:
            out.append(p)
    return out
