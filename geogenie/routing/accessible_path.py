"""Phase 4 accessible routing — Dijkstra/A* stubs + synthetic walk frontier."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from geogenie.core.coords import Origin, to_lonlat, to_xy
from geogenie.routing.graph import Graph

# Preferred walking speed ~1.33 m/s ≈ 80 m/min
WALK_SPEED_M_PER_MIN = 80.0


@dataclass(frozen=True)
class Route:
    nodes: Tuple[int, ...]
    length_m: float
    step_free: bool


def shortest_path(
    graph: Graph,
    src: int,
    dst: int,
    require_step_free: bool = False,
) -> Optional[Route]:
    """Dijkstra shortest path with optional step-free constraint."""
    dist = {src: 0.0}
    prev: dict[int, Optional[int]] = {src: None}
    heap: List[Tuple[float, int]] = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if u == dst:
            break
        for v, w in graph.neighbors(u, require_step_free=require_step_free):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dst not in dist:
        return None
    path = []
    cur: Optional[int] = dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return Route(nodes=tuple(path), length_m=dist[dst], step_free=require_step_free)


def walk_frontier(
    graph: Optional[Graph],
    origin: Origin,
    minutes: float,
    n_samples: int = 36,
) -> List[Tuple[float, float]]:
    """Return frontier points in planar (x, y) metres about origin.

    Until a real street graph is wired, sample a circle at the walk budget.
    When `graph` has node_xy, run a bounded Dijkstra and return reached leaves.
    """
    radius = minutes * WALK_SPEED_M_PER_MIN
    if graph is not None and graph.node_xy and graph.adj:
        return _graph_frontier(graph, origin, radius)

    # Synthetic circular frontier (Phase 4 placeholder)
    pts = []
    for i in range(n_samples):
        ang = 2.0 * math.pi * i / n_samples
        # Slight radial jitter so alpha shape has interior structure
        r = radius * (0.85 + 0.15 * ((i % 5) / 4.0))
        pts.append((r * math.cos(ang), r * math.sin(ang)))
    # Add a few interior samples so Delaunay/alpha is well-defined
    for i in range(8):
        ang = 2.0 * math.pi * i / 8
        pts.append((0.4 * radius * math.cos(ang), 0.4 * radius * math.sin(ang)))
    pts.append((0.0, 0.0))
    return pts


def _graph_frontier(
    graph: Graph, origin: Origin, budget_m: float
) -> List[Tuple[float, float]]:
    """Nodes reachable within budget_m; return their planar coords."""
    # Find nearest graph node to origin
    ox, oy = 0.0, 0.0  # origin projects to (0,0) when Origin is the plane origin
    # Re-project node lon/lat if stored as lonlat — node_xy is already planar
    # about the same origin the caller used.
    best_node, best_d = None, math.inf
    for nid, (x, y) in graph.node_xy.items():
        d = (x - ox) ** 2 + (y - oy) ** 2
        if d < best_d:
            best_d, best_node = d, nid
    if best_node is None:
        return walk_frontier(None, origin, budget_m / WALK_SPEED_M_PER_MIN)

    dist = {best_node: 0.0}
    heap = [(0.0, best_node)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph.neighbors(u):
            nd = d + w
            if nd <= budget_m and nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    # Frontier ≈ nodes with no further expansion within remaining budget
    pts = [graph.node_xy[n] for n in dist if n in graph.node_xy]
    if len(pts) < 3:
        return walk_frontier(None, origin, budget_m / WALK_SPEED_M_PER_MIN)
    return pts
