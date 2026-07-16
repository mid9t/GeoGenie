"""
Shared geometric primitives.

Everything in Phase 2 is built on these three ideas:

  * `cross`       -- the 2D cross product, which answers "which side?".
  * `orientation` -- the sign of that cross product, which is THE decision
                     every hull/polygon algorithm actually makes.
  * distances     -- squared where possible, because sqrt is slow and
                     monotone (if a^2 < b^2 then a < b, so comparisons never
                     need the sqrt at all).

COORDINATE CONVENTION
---------------------
Every function here takes plain (x, y) tuples in a **Euclidean plane**, and
for GeoGenie that means **metres**, not degrees of lat/lon.

This is not pedantry. A degree of latitude is ~111 km everywhere, but a
degree of longitude is ~111 km * cos(latitude): ~85 km in San Francisco,
~0 km at the poles. Feed raw (lon, lat) into these algorithms and the plane
is horizontally squashed, so every distance, every tolerance, and every
Voronoi bisector is wrong by ~24% at SF's latitude.

Project to local metres first, do the geometry, project back for display.

ON ROBUSTNESS (the industrial caveat)
-------------------------------------
`orientation` uses a tolerance to decide "collinear". This is the standard
pragmatic choice and it is *not* exact. With floating point, three points
that are nearly collinear can produce inconsistent answers -- orientation(a,b,c)
and orientation(b,c,a) disagreeing -- which can in principle make a hull
algorithm produce a non-convex result or loop.

Production libraries (CGAL, Shewchuk's predicates, JTS) solve this with
*adaptive exact arithmetic*: compute with floats, and fall back to exact
integer/rational arithmetic only when the result is too close to zero to
trust. That is the genuinely correct answer.

For GPS-derived data in metres, coordinates are ~1e4 with ~1e-16 relative
precision, so the cross product carries absolute error around 1e-8. The
tolerance below sits above that noise floor. Real inputs are never
adversarial enough to break it. Know that the limitation exists, and reach
for exact predicates if you ever ingest gridded/snapped data, where exactly
collinear points are common rather than measure-zero.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]

__all__ = [
    "Point",
    "EPS",
    "cross",
    "orientation",
    "dist2",
    "dist",
    "bbox",
    "signed_area",
    "polygon_area",
    "is_ccw",
    "point_segment_distance",
    "dedupe",
]

# Absolute tolerance for treating a cross product as zero. Chosen to sit just
# above the float64 noise floor for metre-scale coordinates (see module note).
EPS = 1e-9


def cross(o: Point, a: Point, b: Point) -> float:
    """2D cross product of (a - o) x (b - o).

    The single most important function in this file. Its SIGN says which way
    you turn going o -> a -> b:

        > 0  counter-clockwise (left turn)
        < 0  clockwise (right turn)
        = 0  collinear

    Its MAGNITUDE is twice the area of triangle (o, a, b), which is why the
    same routine powers polygon area and point-line distance.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def orientation(o: Point, a: Point, b: Point, eps: float = EPS) -> int:
    """Sign of `cross`, with a tolerance: +1 CCW, -1 CW, 0 collinear."""
    c = cross(o, a, b)
    if c > eps:
        return 1
    if c < -eps:
        return -1
    return 0


def dist2(a: Point, b: Point) -> float:
    """Squared distance. Prefer this for comparisons -- sqrt is monotone, so
    ordering by dist2 equals ordering by dist, at zero cost."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def dist(a: Point, b: Point) -> float:
    return math.sqrt(dist2(a, b))


def bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy). Feed this to a k-d tree range query to prune
    candidates before any expensive polygon test."""
    if not points:
        raise ValueError("bbox of an empty point set is undefined")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def signed_area(poly: Sequence[Point]) -> float:
    """Signed area via the shoelace formula. Positive iff the ring is CCW.

    The sign is the useful part: it is how you detect and fix winding order
    without trusting the caller.
    """
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def polygon_area(poly: Sequence[Point]) -> float:
    return abs(signed_area(poly))


def is_ccw(poly: Sequence[Point]) -> bool:
    return signed_area(poly) > 0.0


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Distance from p to the SEGMENT ab (not the infinite line).

    The clamping matters. Douglas-Peucker is often described with
    "perpendicular distance to the line", but on a segment whose nearest
    point is an endpoint, the perpendicular distance to the infinite line is
    smaller than the true distance -- so an unclamped version under-reports
    the error and drops points it should keep. Clamp.
    """
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        # Degenerate segment: a and b coincide.
        return dist(p, a)
    # Project p onto ab, parameterised t in [0, 1].
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def dedupe(points: Sequence[Point]) -> List[Point]:
    """Remove exact duplicates, preserving order.

    Duplicate points are a leading cause of degenerate behaviour in hull and
    Voronoi code (coincident sites have no well-defined cell), so most entry
    points call this first.
    """
    seen = set()
    out: List[Point] = []
    for p in points:
        t = (p[0], p[1])
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
