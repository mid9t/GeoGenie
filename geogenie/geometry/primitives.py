"""
Shared geometric primitives and predicates.

Everything in this package operates on plain ``(x, y)`` tuples of floats in
**metres** (see projection.py). No classes, no wrappers -- tuples are fast,
hashable, and interoperate with whatever Phase 1 produced.

A note on robustness
--------------------
`orientation` is the predicate that every other algorithm here leans on.
Computed in floating point it can return the *wrong sign* for nearly
collinear points, and a wrong sign can make a convex hull non-convex or send
Bowyer-Watson into an inconsistent state. Production libraries (CGAL,
Shewchuk) use adaptive exact arithmetic for this.

We use a relative epsilon, which is a pragmatic middle ground: it is not
exact, but it degrades gracefully by classifying genuinely ambiguous cases
as collinear rather than picking a sign at random. For synthetic POI data at
metre scale this is comfortably sufficient. It is documented here because
"why is my hull occasionally broken" is a question worth being able to
answer.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]

__all__ = [
    "Point",
    "cross",
    "orientation",
    "dist2",
    "dist",
    "bbox",
    "signed_area",
    "polygon_area",
    "polygon_centroid",
    "is_ccw",
    "ensure_ccw",
    "point_segment_distance",
    "point_segment_distance2",
    "circumcenter",
    "circumradius",
    "dedupe",
]

# Absolute floor for the orientation test, in m^2. Below this the cross
# product is indistinguishable from zero at metre scale.
_ABS_EPS = 1e-10
# Relative tolerance, scaled by the magnitude of the operands.
_REL_EPS = 1e-12


def cross(o: Point, a: Point, b: Point) -> float:
    """2D cross product of (a-o) x (b-o).

    > 0 : b lies left of the ray o->a  (counter-clockwise turn)
    < 0 : b lies right                 (clockwise turn)
    = 0 : collinear
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def orientation(o: Point, a: Point, b: Point) -> int:
    """Sign of `cross`, with a scale-aware tolerance. Returns -1, 0, or +1."""
    d = cross(o, a, b)
    # Scale the tolerance by the size of the terms actually being subtracted,
    # so that coordinates in the hundreds of thousands of metres do not blow
    # past a fixed epsilon.
    scale = (
        abs(a[0] - o[0]) * abs(b[1] - o[1])
        + abs(a[1] - o[1]) * abs(b[0] - o[0])
    )
    eps = _ABS_EPS + _REL_EPS * scale
    if d > eps:
        return 1
    if d < -eps:
        return -1
    return 0


def dist2(a: Point, b: Point) -> float:
    """Squared Euclidean distance. Prefer this in comparisons -- sqrt is a
    monotone transform, so it never changes an ordering, only costs time."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def dist(a: Point, b: Point) -> float:
    return math.sqrt(dist2(a, b))


def bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy)."""
    if not points:
        raise ValueError("bbox() of empty sequence")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def signed_area(poly: Sequence[Point]) -> float:
    """Signed area via the shoelace formula. Positive iff CCW.

    `poly` is a ring given without a repeated closing vertex.
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
    """Unsigned area in m^2."""
    return abs(signed_area(poly))


def polygon_centroid(poly: Sequence[Point]) -> Point:
    """Area centroid (not the vertex mean -- those differ, and the vertex
    mean is biased toward wherever vertices happen to be dense)."""
    n = len(poly)
    if n == 0:
        raise ValueError("centroid of empty polygon")
    if n < 3:
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    a = signed_area(poly)
    if abs(a) < 1e-12:
        # Degenerate (collinear) ring: area centroid is undefined, fall back.
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        c = x1 * y2 - x2 * y1
        cx += (x1 + x2) * c
        cy += (y1 + y2) * c
    return (cx / (6.0 * a), cy / (6.0 * a))


def is_ccw(poly: Sequence[Point]) -> bool:
    return signed_area(poly) > 0.0


def ensure_ccw(poly: Sequence[Point]) -> List[Point]:
    return list(poly) if is_ccw(poly) else list(reversed(poly))


def point_segment_distance2(p: Point, a: Point, b: Point) -> float:
    """Squared distance from p to the *segment* ab (not the infinite line).

    The segment-vs-line distinction matters: Douglas-Peucker is defined on
    the segment, and using line distance would let a far-off point score as
    near because its projection falls outside the segment.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 0.0:
        # Degenerate segment (a == b): fall back to point distance.
        return (px - ax) ** 2 + (py - ay) ** 2
    # Projection parameter of p onto ab, clamped to the segment.
    t = ((px - ax) * vx + (py - ay) * vy) / L2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx, cy = ax + t * vx, ay + t * vy
    return (px - cx) ** 2 + (py - cy) ** 2


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    return math.sqrt(point_segment_distance2(p, a, b))


def circumcenter(a: Point, b: Point, c: Point) -> Point | None:
    """Centre of the circle through a, b, c. None if they are collinear.

    This is the workhorse of both Voronoi (cell vertices are circumcentres of
    Delaunay triangles) and the alpha shape (filtering by circumradius).
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    # Translate to a as origin -- improves conditioning when coordinates are
    # large and the triangle is small, which is exactly our regime.
    bx -= ax
    by -= ay
    cx -= ax
    cy -= ay
    d = 2.0 * (bx * cy - by * cx)
    if abs(d) < 1e-18:
        return None  # collinear
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / d
    uy = (bx * c2 - cx * b2) / d
    return (ax + ux, ay + uy)


def circumradius(a: Point, b: Point, c: Point) -> float:
    """Radius of the circumcircle; math.inf if collinear."""
    cc = circumcenter(a, b, c)
    if cc is None:
        return math.inf
    return dist(cc, a)


def dedupe(points: Sequence[Point], precision: float = 1e-9) -> List[Point]:
    """Remove exact/near-duplicate points, preserving first-seen order.

    Duplicates are not a nuisance here, they are a correctness hazard:
    Bowyer-Watson will produce zero-area triangles from coincident points,
    and a zero-area triangle has no circumcentre.
    """
    seen = set()
    out: List[Point] = []
    inv = 1.0 / precision
    for p in points:
        key = (round(p[0] * inv), round(p[1] * inv))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
