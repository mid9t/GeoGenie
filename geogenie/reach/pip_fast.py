"""Vectorized point-in-polygon — hot path for reach filtering. [VR §4.3]"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from geogenie.geometry.point_in_polygon import is_convex_ring, point_in_convex_polygon

Point = Tuple[float, float]


def pip_filter_vectorized(
    points: np.ndarray,
    ring: np.ndarray,
    include_boundary: bool = True,
) -> np.ndarray:
    """Ray-cast PIP for many points. Returns bool mask of length N.

    points: (N, 2) array
    ring: (M, 2) array, no repeated closing vertex
    """
    points = np.asarray(points, dtype=np.float64)
    ring = np.asarray(ring, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be (N, 2)")
    if len(ring) < 3:
        return np.zeros(len(points), dtype=bool)

    # BBox reject (inflate by boundary eps so near-edge points are considered)
    eps = 1e-3
    pad = eps if include_boundary else 0.0
    minx, miny = ring.min(axis=0)
    maxx, maxy = ring.max(axis=0)
    xs, ys = points[:, 0], points[:, 1]
    inside_box = (
        (xs >= minx - pad)
        & (xs <= maxx + pad)
        & (ys >= miny - pad)
        & (ys <= maxy + pad)
    )
    out = np.zeros(len(points), dtype=bool)
    if not inside_box.any():
        return out

    # Convex fast-path
    ring_list = [tuple(p) for p in ring]
    if is_convex_ring(ring_list):
        return pip_filter_convex_fastpath(points, ring, include_boundary)

    candidates = np.nonzero(inside_box)[0]
    # Vectorized crossing-number over edges
    n = len(ring)
    x = xs[candidates]
    y = ys[candidates]
    crossings = np.zeros(len(candidates), dtype=np.int32)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        # Avoid division by zero when y1 == y2 (horizontal edge): skip
        if y1 == y2:
            continue
        # Edge crosses horizontal ray to +x from point (half-open y intervals)
        cond = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        )
        crossings += cond.astype(np.int32)

    odd = (crossings % 2) == 1
    out[candidates] = odd

    if include_boundary:
        # Mark near-boundary points as inside (1 mm tolerance, metres)
        for idx in candidates:
            if out[idx]:
                continue
            px, py = points[idx]
            for i in range(n):
                ax, ay = ring[i]
                bx, by = ring[(i + 1) % n]
                if _point_seg_dist2(px, py, ax, ay, bx, by) <= eps * eps:
                    out[idx] = True
                    break
    return out


def pip_filter_convex_fastpath(
    points: np.ndarray,
    ring: np.ndarray,
    include_boundary: bool = True,
) -> np.ndarray:
    """O(log n) per point via binary search wedge fan (scalar, ring is convex)."""
    ring_list = [tuple(map(float, p)) for p in ring]
    out = np.zeros(len(points), dtype=bool)
    eps = 1e-3 if include_boundary else 0.0
    minx, miny = ring.min(axis=0)
    maxx, maxy = ring.max(axis=0)
    for i, (x, y) in enumerate(points):
        if x < minx - eps or x > maxx + eps or y < miny - eps or y > maxy + eps:
            continue
        out[i] = point_in_convex_polygon((float(x), float(y)), ring_list, include_boundary)
    return out


def _point_seg_dist2(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        return apx * apx + apy * apy
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    dx, dy = apx - t * abx, apy - t * aby
    return dx * dx + dy * dy


def pip_filter(
    points_xy: Sequence[Point],
    ring_xy: Sequence[Point],
    include_boundary: bool = True,
) -> list[bool]:
    """Convenience wrapper returning a Python list of bools."""
    if not points_xy:
        return []
    mask = pip_filter_vectorized(
        np.asarray(points_xy, dtype=np.float64),
        np.asarray(ring_xy, dtype=np.float64),
        include_boundary=include_boundary,
    )
    return mask.tolist()
