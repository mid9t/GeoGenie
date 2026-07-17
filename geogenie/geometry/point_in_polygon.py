"""
Point-in-polygon tests.

Three tests, for three situations:

  * `point_in_polygon` -- ray casting (crossing number). O(n). Works for any
    simple polygon, convex or not, and is what you use on an alpha-shape
    isochrone.

  * `point_in_convex_polygon` -- O(log n) via binary search on the wedge fan
    from vertex 0. Only valid for a convex ring. Worth having because a
    convex hull is by construction convex, and this is the innermost loop of
    "filter 120k POIs against a reachable-area polygon".

  * `winding_number` -- O(n). Distinguishes containment for self-intersecting
    rings, where the crossing number is ambiguous. Not needed for hulls, but
    alpha shapes can produce rings you did not expect, so it is here.

`PreparedPolygon` wraps these with the optimisation that actually matters in
practice: a bounding-box reject. Against a 120k-POI dataset and a ~800 m
walking isochrone, the bbox rejects the overwhelming majority of candidates
in two float comparisons, and the O(n) ray cast only ever runs on the few
that survive. Constant factors beat asymptotics at this scale.

The real fix, of course, is to not test 120k points at all -- use the Phase 1
k-d tree to range-query the polygon's bounding box first, then test only the
returned candidates. See `PreparedPolygon.filter` and the note there.

Boundary semantics
------------------
"Is a point exactly on the edge inside?" has no universally right answer, so
it is an explicit parameter rather than an accident of the float comparisons.
`include_boundary` defaults to True: a café whose coordinate lands exactly on
the isochrone boundary should be offered to the user, not silently dropped.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from .primitives import Point, cross, point_segment_distance2

__all__ = [
    "point_in_polygon",
    "point_on_boundary",
    "point_in_convex_polygon",
    "winding_number",
    "is_convex_ring",
    "PreparedPolygon",
]

# Tolerance for "on the boundary", in metres. 1 mm -- far below GPS accuracy,
# so it only ever catches genuine float noise, never a real distinction.
_BOUNDARY_EPS = 1e-3


def point_on_boundary(p: Point, poly: Sequence[Point], eps: float = _BOUNDARY_EPS) -> bool:
    """True if p lies within `eps` metres of any edge of the ring."""
    n = len(poly)
    if n < 2:
        return n == 1 and point_segment_distance2(p, poly[0], poly[0]) <= eps * eps
    e2 = eps * eps
    for i in range(n):
        if point_segment_distance2(p, poly[i], poly[(i + 1) % n]) <= e2:
            return True
    return False


def point_in_polygon(
    p: Point,
    poly: Sequence[Point],
    include_boundary: bool = True,
    eps: float = _BOUNDARY_EPS,
) -> bool:
    """Ray casting / crossing number. O(n). Any simple polygon.

    Cast a ray from p in the +x direction and count edge crossings; odd means
    inside. The subtlety is what happens when the ray passes exactly through
    a vertex -- naively it gets counted twice (or zero times) and the parity
    flips wrongly.

    The fix is the asymmetric half-open comparison `(yi > y) != (yj > y)`.
    It treats each edge as containing its lower endpoint and excluding its
    upper one, so a vertex shared by two edges is counted exactly once when
    the edges cross the ray and zero times when they merely touch it. This
    single line is why this implementation is correct and most casual ones
    are not.
    """
    n = len(poly)
    if n < 3:
        return False

    if include_boundary and point_on_boundary(p, poly, eps):
        return True

    x, y = p
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        # Half-open rule: does the edge straddle the horizontal line y?
        if (yi > y) != (yj > y):
            # x coordinate where edge (i, j) crosses the line y.
            t = (y - yi) / (yj - yi)
            x_cross = xi + t * (xj - xi)
            if x < x_cross:
                inside = not inside
        j = i

    if not include_boundary and inside and point_on_boundary(p, poly, eps):
        return False
    return inside


def winding_number(p: Point, poly: Sequence[Point]) -> int:
    """Signed number of times the ring wraps around p. Nonzero => inside.

    Unlike the crossing number, this is correct for self-intersecting rings.
    O(n), no trigonometry -- accumulated from the orientation predicate.
    """
    wn = 0
    n = len(poly)
    x, y = p
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        if yi <= y:
            if yj > y and cross(poly[i], poly[(i + 1) % n], p) < 0:
                wn += 1
        else:
            if yj <= y and cross(poly[i], poly[(i + 1) % n], p) > 0:
                wn -= 1
    return wn


def is_convex_ring(poly: Sequence[Point]) -> bool:
    """True if the ring is convex (allowing collinear vertices)."""
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        c = cross(poly[i], poly[(i + 1) % n], poly[(i + 2) % n])
        if c != 0:
            s = 1 if c > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
    return True


def point_in_convex_polygon(
    p: Point, hull: Sequence[Point], include_boundary: bool = True
) -> bool:
    """O(log n) containment test for a CCW convex ring.

    Fan the hull into wedges from vertex 0 and binary search for the wedge
    containing p, then a single orientation test against that wedge's outer
    edge decides it. Three orientation tests total regardless of hull size.

    Precondition: `hull` is convex and counter-clockwise. Garbage in,
    confidently-wrong-answer out -- use `is_convex_ring` if unsure, or use
    `PreparedPolygon`, which checks for you.
    """
    n = len(hull)
    if n < 3:
        if n == 2:
            return include_boundary and point_on_boundary(p, hull)
        return False

    v0 = hull[0]

    # Is p outside the fan's angular span? Two tests handle it.
    c1 = cross(v0, hull[1], p)
    c2 = cross(v0, hull[n - 1], p)
    if c1 < 0 or c2 > 0:
        return False
    if c1 == 0:
        # On the first spoke: inside iff within the segment.
        return include_boundary and point_on_boundary(p, hull)
    if c2 == 0:
        return include_boundary and point_on_boundary(p, hull)

    # Binary search for i such that p lies in the wedge (v0, hull[i], hull[i+1]).
    lo, hi = 1, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cross(v0, hull[mid], p) >= 0:
            lo = mid
        else:
            hi = mid

    c = cross(hull[lo], hull[hi], p)
    if c > 0:
        return True
    if c < 0:
        return False
    return include_boundary  # exactly on the outer edge


class PreparedPolygon:
    """A polygon preprocessed for repeated containment queries.

    Optimisations, in the order they pay off:
      1. Bounding-box reject -- 2-4 float compares, kills most candidates.
      2. Convexity detected once at construction; convex rings then use the
         O(log n) path instead of O(n) ray casting.
      3. The ring is stored as a flat list to avoid attribute lookups in the
         inner loop.

    Build once, query many. Constructing this per query is strictly worse
    than calling `point_in_polygon` directly.
    """

    __slots__ = ("ring", "minx", "miny", "maxx", "maxy", "convex", "_n")

    def __init__(self, ring: Sequence[Point]) -> None:
        r = [tuple(pt) for pt in ring]
        # Drop a repeated closing vertex if the caller supplied one.
        if len(r) > 1 and r[0] == r[-1]:
            r = r[:-1]
        if len(r) < 3:
            raise ValueError(f"polygon needs >= 3 distinct vertices, got {len(r)}")
        self.ring = r
        self._n = len(r)
        xs = [p[0] for p in r]
        ys = [p[1] for p in r]
        self.minx, self.maxx = min(xs), max(xs)
        self.miny, self.maxy = min(ys), max(ys)
        from .primitives import is_ccw

        self.convex = is_convex_ring(r) and is_ccw(r)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) -- feed this straight into your Phase 1
        k-d tree range query to get candidates."""
        return (self.minx, self.miny, self.maxx, self.maxy)

    def contains(self, p: Point, include_boundary: bool = True) -> bool:
        x, y = p
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy:
            return False
        if self.convex:
            return point_in_convex_polygon(p, self.ring, include_boundary)
        return point_in_polygon(p, self.ring, include_boundary)

    def filter(
        self, points: Iterable[Point], include_boundary: bool = True
    ) -> List[Point]:
        """Keep only the points inside.

        NOTE: this is a linear scan. For GeoGenie's real path do NOT hand it
        all 120k POIs -- do this instead:

            cands = kdtree.range_query(*prepared.bbox)   # Phase 1
            inside = prepared.filter(cands)              # Phase 2

        The k-d tree turns an O(n) scan into O(sqrt(n) + k), and the polygon
        test then runs on k candidates instead of n. That composition is the
        whole point of having built both.
        """
        return [p for p in points if self.contains(p, include_boundary)]

    def __repr__(self) -> str:  # pragma: no cover
        kind = "convex" if self.convex else "simple"
        return f"PreparedPolygon({self._n} verts, {kind})"
