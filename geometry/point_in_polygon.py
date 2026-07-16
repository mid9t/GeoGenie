"""
Point-in-polygon: is this POI inside the reachable area?

TWO ALGORITHMS, DELIBERATELY
----------------------------
`point_in_polygon`        -- ray casting. O(n). Works on ANY simple polygon,
                             convex or not, which matters the moment you move
                             from convex hulls to real isochrone shapes.

`point_in_convex_polygon` -- binary search. O(log n). Only valid on a CONVEX
                             ring, which is exactly what a convex hull gives
                             you. ~5x faster on a 40-vertex hull, and the gap
                             widens with vertex count.

Keep both. The convex one is what you use against a hull; the general one is
what you fall back to when the shape stops being convex, and it doubles as
the oracle you test the fast path against.

RAY CASTING (the Jordan curve theorem, made practical)
------------------------------------------------------
Fire a ray from the query point in any fixed direction (we use +x). Count
edge crossings. Odd = inside, even = outside. That is the whole idea, and it
is beautiful. The difficulty is entirely in the degenerate cases.

THE DEGENERATE CASES (where naive implementations are wrong)
------------------------------------------------------------
The failure mode is a ray that hits a VERTEX exactly. Two edges meet there,
so a careless test counts two crossings where there should be one -- and the
answer flips from inside to outside.

The fix is the asymmetric comparison below:

    (y1 > y) != (y2 > y)

Read it carefully. Each edge is treated as half-open in y: it includes its
lower endpoint and excludes its upper one. So when the ray passes exactly
through a shared vertex, precisely one of the two incident edges counts it.
The double-count disappears. This is why the test uses `>` on one side and
not `>=` -- swapping to `>=` silently reintroduces the bug, and it will not
show up in random testing because the failure has measure zero. It shows up
in production, where coordinates are snapped to grids and vertices line up.

Horizontal edges (y1 == y2) are skipped entirely by the same condition, which
is correct: a ray running along an edge has no well-defined crossing count,
and the edge's endpoints are handled by the neighbouring non-horizontal edges.

BOUNDARY POINTS
---------------
A point exactly on the boundary is genuinely ambiguous -- ray casting will
return an arbitrary answer for it. So we test for it explicitly and let the
caller decide, rather than pretending the ambiguity does not exist. It is a
real question for GeoGenie: is a cafe exactly on the edge of your 10-minute
walk reachable? Default: yes, be generous.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .primitives import EPS, Point, cross, is_ccw, orientation, point_segment_distance

__all__ = [
    "point_in_polygon",
    "point_in_convex_polygon",
    "point_on_boundary",
    "is_convex_ring",
]

# A point within this many metres of an edge counts as "on" it.
# 1e-9 m is a nanometre: this is a float-noise tolerance, not a real distance.
BOUNDARY_EPS = 1e-9


def point_on_boundary(
    p: Point, poly: Sequence[Point], eps: float = BOUNDARY_EPS
) -> bool:
    """True if p lies on any edge of the ring (within `eps`)."""
    n = len(poly)
    for i in range(n):
        if point_segment_distance(p, poly[i], poly[(i + 1) % n]) <= eps:
            return True
    return False


def point_in_polygon(
    p: Point, poly: Sequence[Point], include_boundary: bool = True
) -> bool:
    """Ray casting. Works on any simple polygon (convex or concave). O(n).

    Parameters
    ----------
    p    : (x, y) query point, in the same plane as `poly` (metres).
    poly : ring of >= 3 vertices, open (do not repeat the first vertex).
           Winding order does not matter -- crossing parity is orientation
           independent, which is a nice property of this method.
    include_boundary : what to return for a point exactly on an edge.
    """
    n = len(poly)
    if n < 3:
        return False

    if point_on_boundary(p, poly):
        return include_boundary

    x, y = p
    inside = False
    j = n - 1
    for i in range(n):
        x1, y1 = poly[j]
        x2, y2 = poly[i]

        # Half-open in y: includes the lower endpoint, excludes the upper.
        # This is what makes a ray through a vertex count exactly once.
        # It also skips horizontal edges automatically.
        if (y1 > y) != (y2 > y):
            # x of the intersection between edge (x1,y1)-(x2,y2) and the
            # horizontal line through p. The division is safe: the guard
            # above guarantees y1 != y2.
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:  # crossing is strictly to the right of p
                inside = not inside
        j = i

    return inside


def is_convex_ring(poly: Sequence[Point]) -> bool:
    """True if the ring is convex (all turns the same way, no reflex vertex).

    Cheap O(n) check, so you can decide once whether the O(log n) path is
    legal rather than trusting a caller's claim.
    """
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        o = orientation(poly[i], poly[(i + 1) % n], poly[(i + 2) % n])
        if o == 0:
            continue  # collinear vertices are fine
        if sign == 0:
            sign = o
        elif o != sign:
            return False
    return True


def point_in_convex_polygon(
    p: Point, hull: Sequence[Point], include_boundary: bool = True
) -> bool:
    """Binary search containment for a CONVEX, CCW ring. O(log n).

    PRECONDITION: `hull` is convex and counter-clockwise. Pass a concave ring
    and you get a wrong answer, not an exception. Use `is_convex_ring` if you
    are unsure -- but note that checking costs O(n), which defeats the point
    unless you check once and query many times.

    HOW IT WORKS
    ------------
    Anchor at hull[0] and think of the polygon as a fan of triangles
    (hull[0], hull[i], hull[i+1]). The vertices hull[1..n-1] are sorted by
    angle around hull[0] -- that is exactly what convexity buys us -- so we
    can BINARY SEARCH for the fan wedge containing p, then do one final
    orientation test against that wedge's outer edge.

      O(log n) to find the wedge + O(1) to test it.
    """
    n = len(hull)
    if n < 3:
        return False

    a = hull[0]

    # Is p outside the fan's two bounding rays? Then it is outside, cheaply.
    o_first = orientation(a, hull[1], p)
    o_last = orientation(a, hull[n - 1], p)
    if o_first < 0 or o_last > 0:
        return False
    # p lies exactly along a bounding ray: it is inside only if it is also
    # within the segment, which the boundary test below resolves.
    if o_first == 0:
        return (
            include_boundary
            if point_segment_distance(p, a, hull[1]) <= BOUNDARY_EPS
            else _between(a, hull[1], p)
        )
    if o_last == 0:
        return (
            include_boundary
            if point_segment_distance(p, a, hull[n - 1]) <= BOUNDARY_EPS
            else _between(a, hull[n - 1], p)
        )

    # Binary search for the wedge: largest lo with p left-of-or-on (a, hull[lo]).
    lo, hi = 1, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if orientation(a, hull[mid], p) >= 0:
            lo = mid
        else:
            hi = mid

    # One last test against the outer edge of the wedge.
    o = orientation(hull[lo], hull[hi], p)
    if o == 0:
        return include_boundary
    return o > 0


def _between(a: Point, b: Point, p: Point) -> bool:
    """p is collinear with a,b -- is it within the segment's extent?"""
    return (
        min(a[0], b[0]) - BOUNDARY_EPS <= p[0] <= max(a[0], b[0]) + BOUNDARY_EPS
        and min(a[1], b[1]) - BOUNDARY_EPS <= p[1] <= max(a[1], b[1]) + BOUNDARY_EPS
    )
