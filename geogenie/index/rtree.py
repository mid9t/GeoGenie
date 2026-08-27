"""
STR (Sort-Tile-Recursive) bulk-loaded R-tree behind SpatialIndex.

Dataset is static-at-startup (DB is truth, index rebuilt on boot), so
incremental insert/split machinery has no payoff — STR packs near-optimally
in O(n log n). Dynamic inserts would want linear/quadratic split or R*.

Circle range_query uses bbox-circle intersection (clamp center to bbox).
k_nearest is best-first search (Hjaltason–Samet) on mindist(query, bbox).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import POI
from geogenie.index._dist import (
    circle_intersects_bbox,
    mindist_sq_to_bbox,
    sq_dist,
    sq_dist_to_points,
)

BBox = Tuple[float, float, float, float]  # minx, miny, maxx, maxy


@dataclass
class _Node:
    bbox: BBox
    children: Optional[List["_Node"]] = None
    # leaf entries: rows of (x, y, id)
    entries: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return self.entries is not None


def _bbox_of_points(xy: np.ndarray) -> BBox:
    return (
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    )


def _bbox_of_nodes(nodes: Sequence[_Node]) -> BBox:
    minx = min(n.bbox[0] for n in nodes)
    miny = min(n.bbox[1] for n in nodes)
    maxx = max(n.bbox[2] for n in nodes)
    maxy = max(n.bbox[3] for n in nodes)
    return (minx, miny, maxx, maxy)


def _bbox_center(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


class RTreeIndex:
    """SpatialIndex via STR bulk-loaded R-tree. Returns ids."""

    def __init__(
        self,
        origin: Optional[Origin] = None,
        leaf_capacity: int = 32,
        fanout: int = 16,
    ) -> None:
        self.origin: Optional[Origin] = origin
        self.leaf_capacity = max(2, leaf_capacity)
        self.fanout = max(2, fanout)
        self.root: Optional[_Node] = None
        self._by_id: Dict[int, POI] = {}
        self._pois: List[POI] = []
        self._n = 0

    def build(self, pois: list[POI]) -> None:
        if not pois:
            self.root = None
            self._by_id = {}
            self._pois = []
            self._n = 0
            return
        if self.origin is None:
            lon0 = sum(p.lon for p in pois) / len(pois)
            lat0 = sum(p.lat for p in pois) / len(pois)
            self.origin = Origin(lon0, lat0)
        projected: List[POI] = []
        rows = []
        for p in pois:
            x, y = to_xy(p.lon, p.lat, self.origin)
            projected.append(replace(p, x=x, y=y))
            rows.append((x, y, float(p.id)))
        self._pois = projected
        self._by_id = {p.id: p for p in projected}
        self._n = len(projected)
        entries = np.asarray(rows, dtype=np.float64)
        self.root = self._str_bulk_load(entries, is_leaf_level=True)

    def get_pois(self, ids: Sequence[int]) -> List[POI]:
        return [self._by_id[i] for i in ids if i in self._by_id]

    def _str_bulk_load(self, items: np.ndarray, is_leaf_level: bool) -> _Node:
        """STR: sort by x into strips, sort each strip by y, pack into nodes."""
        n = len(items)
        if n == 0:
            empty = np.empty((0, 3), dtype=np.float64)
            return _Node(bbox=(0.0, 0.0, 0.0, 0.0), entries=empty)

        capacity = self.leaf_capacity if is_leaf_level else self.fanout

        if is_leaf_level:
            # items are (x, y, id)
            if n <= capacity:
                return _Node(bbox=_bbox_of_points(items[:, :2]), entries=items.copy())

            n_leaves = math.ceil(n / capacity)
            n_slices = max(1, math.ceil(math.sqrt(n_leaves)))
            # sort by x
            order = np.argsort(items[:, 0], kind="mergesort")
            sorted_items = items[order]
            slice_size = math.ceil(n / n_slices)
            leaves: List[_Node] = []
            for s in range(n_slices):
                strip = sorted_items[s * slice_size : (s + 1) * slice_size]
                if len(strip) == 0:
                    continue
                y_order = np.argsort(strip[:, 1], kind="mergesort")
                strip = strip[y_order]
                for i in range(0, len(strip), capacity):
                    chunk = strip[i : i + capacity]
                    leaves.append(
                        _Node(bbox=_bbox_of_points(chunk[:, :2]), entries=chunk.copy())
                    )
            return self._pack_nodes(leaves)

        # Internal level: items are node centers packed as (cx, cy, node_idx)
        # We don't use this path with raw arrays — _pack_nodes handles internals.
        raise RuntimeError("internal STR path goes through _pack_nodes")

    def _pack_nodes(self, nodes: List[_Node]) -> _Node:
        """Recursively STR-pack a list of child nodes into a tree."""
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) <= self.fanout:
            return _Node(bbox=_bbox_of_nodes(nodes), children=list(nodes))

        # Represent each node by its bbox center for tiling
        centers = np.array([_bbox_center(n.bbox) for n in nodes], dtype=np.float64)
        n = len(nodes)
        n_groups = math.ceil(n / self.fanout)
        n_slices = max(1, math.ceil(math.sqrt(n_groups)))
        order = np.argsort(centers[:, 0], kind="mergesort")
        slice_size = math.ceil(n / n_slices)
        packed: List[_Node] = []
        for s in range(n_slices):
            idxs = order[s * slice_size : (s + 1) * slice_size]
            if len(idxs) == 0:
                continue
            strip_nodes = [nodes[i] for i in idxs]
            strip_centers = centers[idxs]
            y_order = np.argsort(strip_centers[:, 1], kind="mergesort")
            strip_nodes = [strip_nodes[i] for i in y_order]
            for i in range(0, len(strip_nodes), self.fanout):
                chunk = strip_nodes[i : i + self.fanout]
                packed.append(_Node(bbox=_bbox_of_nodes(chunk), children=chunk))
        return self._pack_nodes(packed)

    def range_query(self, x: float, y: float, radius_m: float) -> list[int]:
        if self.root is None:
            return []
        r2 = radius_m * radius_m
        found: List[int] = []

        def _search(node: _Node) -> None:
            minx, miny, maxx, maxy = node.bbox
            if not circle_intersects_bbox(x, y, r2, minx, miny, maxx, maxy):
                return
            if node.is_leaf:
                assert node.entries is not None
                if len(node.entries) == 0:
                    return
                d2 = sq_dist_to_points(x, y, node.entries[:, :2])
                hits = node.entries[d2 <= r2, 2]
                found.extend(int(i) for i in hits)
            else:
                assert node.children is not None
                for child in node.children:
                    _search(child)

        _search(self.root)
        return found

    def k_nearest(self, x: float, y: float, k: int) -> list[int]:
        """Best-first kNN (Hjaltason–Samet): heap on mindist to bbox / point."""
        if k <= 0 or self.root is None or self._n == 0:
            return []
        k = min(k, self._n)
        # heap entries: (dist_sq, tie, kind, payload)
        # kind 0 = node, 1 = point; payload = node or (id,)
        tie = 0
        heap: List[Tuple[float, int, int, object]] = []
        root = self.root
        heapq.heappush(
            heap,
            (mindist_sq_to_bbox(x, y, *root.bbox), tie, 0, root),
        )
        tie += 1
        results: List[int] = []
        while heap and len(results) < k:
            dist, _, kind, payload = heapq.heappop(heap)
            if kind == 1:
                results.append(int(payload))  # type: ignore[arg-type]
                continue
            node = payload  # type: ignore[assignment]
            assert isinstance(node, _Node)
            if node.is_leaf:
                assert node.entries is not None
                for row in node.entries:
                    px, py, pid = float(row[0]), float(row[1]), int(row[2])
                    heapq.heappush(
                        heap, (sq_dist(px, py, x, y), tie, 1, pid)
                    )
                    tie += 1
            else:
                assert node.children is not None
                for child in node.children:
                    md = mindist_sq_to_bbox(x, y, *child.bbox)
                    heapq.heappush(heap, (md, tie, 0, child))
                    tie += 1
        return results
