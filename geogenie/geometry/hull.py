"""
Convex hull.

Two implementations, both O(n log n) and both dominated by the sort:

  * `monotone_chain` (Andrew, 1979) -- sorts lexicographically by (x, y) and
    builds the lower and upper chains. This is the default. It has no polar
    angle computation, no pivot, no special-casing of collinear runs, and no
    trigonometry. Fewer moving parts means fewer places to be subtly wrong.

  * `graham_scan` (1972) -- sorts by polar angle about the bottom-most pivot.
    Included because it is the canonical textbook answer and you asked for
    it, and because being able to say *why* you'd reach for the other one is
    the more interesting version of that interview answer.

Both return a ring in counter-clockwise order with no repeated closing
vertex, and by default contain only strictly convex vertices (points lying
*on* a hull edge are dropped).

------------------------------------------------------------------------
A caveat about using this for "reachable area", which is what Phase 2 asks
------------------------------------------------------------------------
The convex hull of a set of walking-reachable points is an *over-estimate*
of the reachable region, and often a badly wrong one. Real isochrones are
non-convex: they are notched by rivers, railways, motorways, and dead-end
streets. Take the reachable points on both banks of a river with a bridge
1 km upstream -- the convex hull spans the water, and every POI on the far
bank falls inside a polygon it takes twenty minutes to actually reach.

The hull is fast and it is a fine *conservative prefilter* -- it never
excludes a truly reachable point, so it is safe as stage one of a funnel
that later verifies with real network routing. It is not safe as a final
answer. For a shape that actually tracks the reachable set, use
`concave.alpha_shape`, which is in this package and is built on the same
Delaunay triangulation the Voronoi code needs anyway.

Recommended pipeline: hull (cheap, conservative) -> alpha shape (tight) ->
network routing (truth, Phase 3).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .primitives import Point, cross, dist2, polygon_area

__all__ = ["convex_hull", "monotone_chain", "graham_scan", "hull_diameter"]


def monotone_chain(points: Sequence[Point], include_collinear: bool = False) -> List[Point]:
    """Andrew's monotone chain. Returns a CCW ring, no closing vertex.

    include_collinear=True keeps vertices that lie exactly on a hull edge.
    Default is False, which yields the minimal representation.
    """
    # Sorting also deduplicates, which matters: coincident points would
    # otherwise produce zero-length edges with an undefined turn direction.
    pts = sorted(set(map(tuple, points)))
    n = len(pts)
    if n <= 2:
        return list(pts)

    # With include_collinear, we must NOT pop on a zero cross product.
    def bad(o: Point, a: Point, b: Point) -> bool:
        c = cross(o, a, b)
        return c < 0 or (c == 0 and not include_collinear)

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and bad(lower[-2], lower[-1], p):
            lower.pop()
        lower.append(p)

    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and bad(upper[-2], upper[-1], p):
            upper.pop()
        upper.append(p)

    # Drop each chain's last point: it is the other chain's first point.
    hull = lower[:-1] + upper[:-1]

    # Degenerate case: all input points collinear. Both chains trace the same
    # segment and the concatenation doubles back on itself.
    if len(hull) == 2 and hull[0] == hull[1]:
        return [hull[0]]
    return hull


def graham_scan(points: Sequence[Point]) -> List[Point]:
    """Graham scan. Returns a CCW ring, no closing vertex, strict vertices only.

    The classic bug in this algorithm is the handling of points collinear
    with the pivot: if several share a polar angle, the ones in the *final*
    angular group must be visited farthest-first or the scan walks into the
    interior and strands them on the hull. We sidestep it by collapsing each
    angular group to its farthest member before scanning -- those are the
    only candidates that can be hull vertices anyway.
    """
    pts = list(set(map(tuple, points)))
    if len(pts) <= 2:
        return sorted(pts)

    # Pivot: lowest y, then lowest x. Guaranteed to be a hull vertex.
    pivot = min(pts, key=lambda p: (p[1], p[0]))
    rest = [p for p in pts if p != pivot]
    if not rest:
        return [pivot]

    # Sort by polar angle about the pivot. atan2 would work but is slower and
    # introduces float error; since every point is at or above the pivot, the
    # angles lie in [0, pi] and the orientation predicate orders them directly.
    def angle_key(p: Point):
        import math

        return (math.atan2(p[1] - pivot[1], p[0] - pivot[0]), dist2(pivot, p))

    rest.sort(key=angle_key)

    # Collapse collinear-with-pivot groups to the farthest point.
    pruned: List[Point] = []
    for p in rest:
        if pruned and cross(pivot, pruned[-1], p) == 0:
            # Same ray from the pivot; keep whichever is farther.
            if dist2(pivot, p) > dist2(pivot, pruned[-1]):
                pruned[-1] = p
        else:
            pruned.append(p)

    if len(pruned) < 2:
        return [pivot] + pruned

    stack: List[Point] = [pivot, pruned[0]]
    for p in pruned[1:]:
        # Pop while the last turn is not a strict left turn.
        while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)
    return stack


def convex_hull(points: Sequence[Point], include_collinear: bool = False) -> List[Point]:
    """Default entry point. Uses monotone chain."""
    return monotone_chain(points, include_collinear=include_collinear)


def hull_diameter(hull: Sequence[Point]) -> Tuple[float, Point, Point]:
    """Farthest pair of points on a convex ring, via rotating calipers.

    O(n) given the hull, so O(n log n) overall -- versus O(n^2) for the
    brute-force pair scan. Useful for GeoGenie as a fast "how big is this
    reachable area, really" summary statistic, and as a sanity check that a
    candidate isochrone is not absurdly large.

    Returns (distance, point_a, point_b).
    """
    n = len(hull)
    if n < 2:
        raise ValueError("diameter needs at least 2 points")
    if n == 2:
        from .primitives import dist

        return (dist(hull[0], hull[1]), hull[0], hull[1])

    best = -1.0
    best_pair = (hull[0], hull[1])
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        # Advance the opposing caliper while the triangle area grows -- area
        # is a proxy for perpendicular distance from edge (i, i+1).
        while True:
            nj = (j + 1) % n
            if abs(cross(hull[i], hull[ni], hull[nj])) > abs(
                cross(hull[i], hull[ni], hull[j])
            ):
                j = nj
            else:
                break
        for cand in (hull[i], hull[ni]):
            d = dist2(cand, hull[j])
            if d > best:
                best = d
                best_pair = (cand, hull[j])
    import math

    return (math.sqrt(best), best_pair[0], best_pair[1])


def hull_area(points_or_hull: Sequence[Point], already_hull: bool = False) -> float:
    """Area in m^2 of the convex hull of the input."""
    ring = points_or_hull if already_hull else convex_hull(points_or_hull)
    return polygon_area(ring)
