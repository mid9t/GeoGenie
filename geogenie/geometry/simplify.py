"""
Polyline simplification.

`douglas_peucker` is the one you asked for and the one real Maps pipelines
use (it is what powers encoded-polyline zoom levels, GPS trace compression,
and vector-tile generalisation).

Two implementation decisions worth defending:

1. **Iterative, not recursive.** The textbook version recurses. On a real GPS
   trace -- tens of thousands of points, and pathologically on a track that
   is already nearly a straight line -- the recursion depth approaches n and
   Python blows its ~1000-frame stack. The explicit-stack version below is
   the same algorithm with the same output and no depth limit. A recursive
   implementation is not a smaller bug, it is a latent crash that only fires
   on your biggest input.

2. **Squared distances throughout.** The comparison `d > tol` is equivalent
   to `d^2 > tol^2` for non-negative d, so every sqrt in the inner loop is
   removable. That loop runs O(n log n) times on average.

Complexity: O(n log n) expected, O(n^2) worst case (a trace where every split
peels off a single point). The worst case is real but rare on GPS data.

Tolerance is in **metres**, which is only meaningful because the input is
projected (see projection.py). A tolerance in degrees would mean different
things on the x and y axes -- a nonsense unit.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from .primitives import Point, dist2, point_segment_distance2

__all__ = [
    "douglas_peucker",
    "douglas_peucker_mask",
    "radial_distance_filter",
    "simplify",
    "visvalingam_whyatt",
]


def douglas_peucker_mask(points: Sequence[Point], tolerance: float) -> List[bool]:
    """Return a keep/drop mask rather than the points themselves.

    Useful when each vertex carries a payload (timestamp, speed, heading)
    that must be carried along with the geometry -- you simplify the geometry
    and index the payload with the same mask, instead of trying to match
    coordinates back up afterwards.
    """
    n = len(points)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if n <= 2:
        return [True] * n

    keep = [False] * n
    keep[0] = keep[n - 1] = True
    tol2 = tolerance * tolerance

    # Explicit stack of index ranges still to be examined.
    stack: List[tuple[int, int]] = [(0, n - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue  # nothing between the endpoints

        a, b = points[first], points[last]
        max_d2 = -1.0
        idx = -1
        for i in range(first + 1, last):
            d2 = point_segment_distance2(points[i], a, b)
            if d2 > max_d2:
                max_d2 = d2
                idx = i

        if max_d2 > tol2:
            # This vertex is essential: keep it and recurse into both halves.
            keep[idx] = True
            stack.append((first, idx))
            stack.append((idx, last))
        # else: every vertex strictly between first and last is within
        # tolerance of the chord, so all of them are dropped at once.

    return keep


def douglas_peucker(points: Sequence[Point], tolerance: float) -> List[Point]:
    """Simplify a polyline so no dropped vertex is more than `tolerance`
    metres from the retained line. Endpoints are always preserved."""
    if len(points) <= 2:
        return list(points)
    mask = douglas_peucker_mask(points, tolerance)
    return [p for p, k in zip(points, mask) if k]


def radial_distance_filter(points: Sequence[Point], tolerance: float) -> List[Point]:
    """Drop points within `tolerance` metres of the previously kept point.

    O(n), and a very effective *prefilter* for Douglas-Peucker on raw GPS:
    a stationary receiver emits a dense cloud of near-identical fixes, and
    those contribute nothing but O(n^2) risk to DP. Stripping them first
    typically cuts DP's input substantially for near-zero cost.

    This is a cruder criterion than DP -- it can drop a vertex DP would keep,
    so it is lossy in a way DP is not. Use it only with a tolerance well
    below the DP tolerance (a fifth is a reasonable rule of thumb).
    """
    n = len(points)
    if n <= 2:
        return list(points)
    tol2 = tolerance * tolerance
    out = [points[0]]
    for i in range(1, n - 1):
        if dist2(points[i], out[-1]) > tol2:
            out.append(points[i])
    out.append(points[n - 1])
    return out


def simplify(
    points: Sequence[Point], tolerance: float, high_quality: bool = False
) -> List[Point]:
    """Convenience wrapper: optional radial prefilter, then Douglas-Peucker.

    high_quality=True skips the prefilter and runs DP on the raw input.
    """
    if len(points) <= 2:
        return list(points)
    pts = points if high_quality else radial_distance_filter(points, tolerance / 5.0)
    return douglas_peucker(pts, tolerance)


def visvalingam_whyatt(points: Sequence[Point], target: int) -> List[Point]:
    """Simplify to exactly `target` vertices by repeatedly removing the
    vertex whose triangle with its neighbours has the smallest area.

    Included as the counterpart to DP because the two answer different
    questions, and knowing which you want is the actual skill:

      * Douglas-Peucker takes a *tolerance* and gives you however many points
        it takes to honour it. Use it when correctness has a metric bound
        ("never deviate from the true path by more than 5 m").
      * Visvalingam-Whyatt takes a *budget* and gives you the visually least
        damaging polyline of that size. Use it when the constraint is the
        wire or the screen ("this route must fit in 200 points").

    VW also tends to look better at aggressive simplification: DP preserves
    spikes (a spike is by definition far from the chord), while VW removes
    them (a spike has a thin triangle). For GeoGenie, DP for correctness-
    critical geometry, VW for a route drawn on a map at low zoom.

    This implementation is O(n^2) via a linear min-scan; a heap with lazy
    deletion brings it to O(n log n). n is small in the display path, so the
    simple version stands until a profile says otherwise.
    """
    n = len(points)
    if target >= n or n <= 2:
        return list(points)
    if target < 2:
        raise ValueError("target must be >= 2")

    alive = list(range(n))

    def tri_area(ia: int, ib: int, ic: int) -> float:
        ax, ay = points[ia]
        bx, by = points[ib]
        cx, cy = points[ic]
        return abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2.0

    while len(alive) > target:
        best_area = math.inf
        best_pos = -1
        for pos in range(1, len(alive) - 1):  # never remove endpoints
            a = tri_area(alive[pos - 1], alive[pos], alive[pos + 1])
            if a < best_area:
                best_area = a
                best_pos = pos
        if best_pos < 0:
            break
        alive.pop(best_pos)

    return [points[i] for i in alive]
