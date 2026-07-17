"""
Reachability composition: index prefilter → alpha ring → exact PIP.

    radius = minutes * WALK_SPEED * SLACK
    candidates = index.range_query(...)
    ring = ring_cache.get_or_build(...)
    survivors = pip_filter(candidates, ring)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geogenie.core.coords import Origin, to_lonlat, to_xy
from geogenie.core.types import POI, ReachResult, ReachRing
from geogenie.geometry.concave import reachable_area
from geogenie.reach.pip_fast import pip_filter_vectorized
from geogenie.reach.ring_cache import RingCache
from geogenie.routing.accessible_path import WALK_SPEED_M_PER_MIN, walk_frontier
from geogenie.routing.graph import Graph

# Slack > 1 so the circle over-approximates network distance (admissible).
SLACK = 1.25


def _build_ring(
    origin: Origin,
    minutes: float,
    walk_graph: Optional[Graph] = None,
    method: str = "alpha",
) -> ReachRing:
    frontier_xy = walk_frontier(walk_graph, origin, minutes)
    ring_xy = reachable_area(frontier_xy, method=method)
    if len(ring_xy) < 3:
        # Degenerate — fall back to convex sample of the circle
        ring_xy = reachable_area(frontier_xy, method="convex")
    ring_lonlat = tuple(to_lonlat(x, y, origin) for x, y in ring_xy)
    return ReachRing(
        ring_lonlat=ring_lonlat,
        origin_lon=origin.lon,
        origin_lat=origin.lat,
        minutes=minutes,
        method=method,
    )


def _apply_filters(pois: Sequence[POI], filters: Optional[Dict[str, Any]]) -> List[POI]:
    if not filters:
        return list(pois)
    out = []
    for p in pois:
        if "accessible" in filters and filters["accessible"] is not None:
            if bool(p.accessible) != bool(filters["accessible"]):
                continue
        if "noise_level" in filters and filters["noise_level"] is not None:
            if p.noise_level != filters["noise_level"]:
                continue
        if "category" in filters and filters["category"] is not None:
            if p.category != filters["category"]:
                continue
        out.append(p)
    return out


def reachable_pois(
    origin: Origin,
    minutes: float,
    index,
    walk_graph: Optional[Graph] = None,
    ring_cache: Optional[RingCache] = None,
    filters: Optional[Dict[str, Any]] = None,
    method: str = "alpha",
) -> ReachResult:
    """Full reachability pipeline. Index must already be built in planar metres."""
    if minutes <= 0:
        raise ValueError(f"minutes must be positive, got {minutes}")

    radius = minutes * WALK_SPEED_M_PER_MIN * SLACK
    # Project query into the index's plane if the index has a fixed origin;
    # otherwise use the query origin (tree must have been built with same origin).
    idx_origin = getattr(index, "origin", None) or origin
    qx, qy = to_xy(origin.lon, origin.lat, idx_origin)

    candidates = index.range_query(qx, qy, radius)

    cache = ring_cache or RingCache()
    ring = cache.get_or_build(
        origin,
        minutes,
        lambda o, m: _build_ring(o, m, walk_graph=walk_graph, method=method),
    )

    # PIP in the query-origin plane for the ring; candidates need coords in same plane.
    ring_xy = [to_xy(lon, lat, origin) for lon, lat in ring.ring_lonlat]
    if candidates:
        pts = np.array(
            [to_xy(p.lon, p.lat, origin) for p in candidates], dtype=np.float64
        )
        mask = pip_filter_vectorized(pts, np.asarray(ring_xy, dtype=np.float64))
        survivors = [p for p, keep in zip(candidates, mask) if keep]
    else:
        survivors = []

    survivors = _apply_filters(survivors, filters)

    return ReachResult(
        pois=tuple(survivors),
        ring=ring,
        candidates=len(candidates),
        survivors=len(survivors),
        stats={
            "radius_m": radius,
            "method": method,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
        },
    )
