"""
Douglas-Peucker line simplification -- thin out a raw GPS track.

THE PROBLEM
-----------
A phone emits a GPS fix every second. A 30-minute walk is ~1800 points, most
of which encode nothing but receiver noise. Sending them to a browser, or
drawing them, or storing them, wastes bandwidth and time. But you cannot just
drop every other point -- that destroys corners, which are the only part of a
route anyone cares about.

Douglas-Peucker keeps the points that carry SHAPE and discards the ones that
sit on a line between their neighbours.

THE ALGORITHM
-------------
Recursive divide and conquer:

  1. Draw a chord from the first point to the last.
  2. Find the point farthest from that chord.
  3. If its distance <= tolerance, every intermediate point is within
     tolerance of the chord, so discard them all and keep just the endpoints.
  4. Otherwise that point is structurally essential: keep it, split the track
     there, and recurse on both halves.

THE GUARANTEE (this is what makes it the industry default)
----------------------------------------------------------
Every discarded point lies within `tolerance` of the simplified line. That is
a real, checkable bound, not a heuristic -- which is why you can say
"simplified to 5 m accuracy" and mean it. Pick the tolerance from your actual
error budget: GPS is ~5 m accurate, so simplifying below that is preserving
noise; a tolerance near your display's pixel size is the usual choice.

COMPLEXITY
----------
  Average: O(n log n) -- balanced splits.
  Worst:   O(n^2)     -- when each split peels off one point, e.g. a track
                         shaped like a monotone spiral.
  Space:   O(n) for the mask, plus O(log n) average stack.

WHY THIS IMPLEMENTATION IS ITERATIVE
------------------------------------
The recursive form is prettier and appears in every textbook. It also blows
Python's default 1000-frame stack on a pathological ~1000-point track, in
production, on a long ride, at 3am. An explicit stack costs three extra lines
and removes an entire class of incident. Textbook recursion is a teaching
device; an explicit stack is what ships.

COORDINATES
-----------
(x, y) in METRES. `tolerance` is a distance in the same units.

Simplifying raw (lon, lat) degrees is a real and common bug: a tolerance of
1e-4 degrees means ~11 m north-south but only ~8.5 m east-west at 40N, so
your simplification is anisotropic and its meaning drifts with latitude.
Project to metres first.
"""

from __future__ import annotations

from typing import List, Sequence

from .primitives import Point, point_segment_distance

__all__ = ["douglas_peucker", "douglas_peucker_mask", "radial_distance_filter"]


def douglas_peucker_mask(points: Sequence[Point], tolerance: float) -> List[bool]:
    """Which points survive? Returns a keep/drop mask parallel to `points`.

    Exposed separately from `douglas_peucker` because real pipelines carry
    per-point payload -- timestamps, speed, elevation, accuracy -- and you
    need to filter those arrays in lockstep with the coordinates. Returning
    only the surviving coordinates throws away the indices you need to do
    that, and callers end up doing an O(n*m) re-match to recover them.
    """
    n = len(points)
    if n == 0:
        return []
    if n <= 2:
        return [True] * n
    if tolerance <= 0:
        # A zero tolerance means "keep everything with any deviation at all".
        return [True] * n

    keep = [False] * n
    keep[0] = True
    keep[n - 1] = True

    # Explicit stack of (first, last) index ranges. See module note.
    stack: List[tuple] = [(0, n - 1)]

    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue  # nothing between them

        a = points[first]
        b = points[last]

        # Find the point farthest from the chord a->b.
        dmax = -1.0
        idx = first
        for i in range(first + 1, last):
            d = point_segment_distance(points[i], a, b)
            if d > dmax:
                dmax = d
                idx = i

        if dmax > tolerance:
            keep[idx] = True
            # Recurse into both halves. Order does not matter for the result.
            stack.append((first, idx))
            stack.append((idx, last))
        # else: every intermediate point is within tolerance of the chord,
        # so all of them are dropped and there is nothing to recurse into.

    return keep


def douglas_peucker(points: Sequence[Point], tolerance: float) -> List[Point]:
    """Simplified polyline. Endpoints are always preserved.

    The output is always a SUBSET of the input -- no points are invented or
    moved. That matters: it means a simplified track still consists of places
    the user actually was.
    """
    mask = douglas_peucker_mask(points, tolerance)
    return [p for p, k in zip(points, mask) if k]


def radial_distance_filter(points: Sequence[Point], tolerance: float) -> List[Point]:
    """Drop points closer than `tolerance` to the previously kept point. O(n).

    Not a replacement for Douglas-Peucker -- it has no error guarantee and
    will happily cut a corner. It is a cheap PRE-filter for one specific
    pathology: a stationary receiver. Stand still at a traffic light for 60
    seconds and GPS emits 60 points scattered in a ~5 m blob. Douglas-Peucker
    handles them correctly but pays O(k) distance computations per point to
    do it; this strips the blob to one point in a single linear pass first.

    Typical use: radial_distance_filter(track, 2.0) then douglas_peucker(..., 5.0)
    """
    n = len(points)
    if n <= 2 or tolerance <= 0:
        return list(points)

    tol2 = tolerance * tolerance
    out = [points[0]]
    for i in range(1, n - 1):
        px, py = points[i]
        qx, qy = out[-1]
        dx = px - qx
        dy = py - qy
        if dx * dx + dy * dy >= tol2:
            out.append(points[i])
    out.append(points[n - 1])  # last point is never dropped
    return out
