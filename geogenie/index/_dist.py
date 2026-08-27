"""Single squared-distance predicate shared by all indices and the oracle.

Two distance implementations is how set-equality gates start flaking.
"""

from __future__ import annotations

import numpy as np


def sq_dist(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def sq_dist_to_points(x: float, y: float, pts: np.ndarray) -> np.ndarray:
    """Squared distances from (x, y) to each row of pts (N, 2)."""
    d = pts - np.array([x, y], dtype=np.float64)
    return d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]


def circle_intersects_bbox(
    x: float, y: float, r2: float, minx: float, miny: float, maxx: float, maxy: float
) -> bool:
    """True if circle centered at (x,y) with squared radius r2 intersects bbox.

    Clamp center to bbox, then check squared distance — tighter than bbox-bbox
    against the circle's bounding square.
    """
    cx = min(max(x, minx), maxx)
    cy = min(max(y, miny), maxy)
    return sq_dist(x, y, cx, cy) <= r2


def mindist_sq_to_bbox(
    x: float, y: float, minx: float, miny: float, maxx: float, maxy: float
) -> float:
    """Minimum squared Euclidean distance from point to axis-aligned bbox."""
    dx = 0.0 if minx <= x <= maxx else (minx - x if x < minx else x - maxx)
    dy = 0.0 if miny <= y <= maxy else (miny - y if y < miny else y - maxy)
    return dx * dx + dy * dy
