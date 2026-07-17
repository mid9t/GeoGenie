"""
Concave hull (alpha shape) -- the reachable-area polygon that is actually
shaped like a reachable area.

The problem with the convex hull
--------------------------------
Phase 2 asks for a convex hull to bound the walking-reachable region. It will
run, and it will over-claim. A convex hull cannot express a notch, and every
real isochrone is nothing but notches: rivers, rail cuttings, motorways,
private land, cul-de-sacs. Reachable points on both banks of a river whose
nearest bridge is a kilometre upstream produce a hull that calmly spans the
water and declares everything between them reachable in ten minutes.

That is not a rounding error. It is the difference between "quiet café, 8
minute walk" and "quiet café, 25 minute walk, wrong side of the river" -- and
for the accessibility use case in the one-liner, sending a wheelchair user on
a route that does not exist is the worst failure this system can produce.

The fix
-------
An **alpha shape**. Take the Delaunay triangulation of the reachable points
and keep only triangles whose circumcircle is smaller than a threshold. The
intuition: a triangle spanning a river is huge and empty, so its circumcircle
is huge. A triangle inside a dense block of reachable points is small. Filter
by circumradius and the empty spaces fall out on their own. The boundary of
what survives is a concave polygon that hugs the actual point set.

alpha = infinity recovers the convex hull exactly; alpha -> 0 erodes the
shape to nothing. In between is a knob, and choosing it is a judgement call --
`suggest_alpha` gives a defensible data-driven default.

Where this sits in the funnel
-----------------------------
Still an approximation. It is a shape fitted to *sampled* reachable points, so
it is only as good as the sampling, and it knows nothing about one-way
streets, stairs, or opening hours. It is a much tighter prefilter than the
hull -- and unlike the hull it can produce false negatives, so if you need
the guarantee that nothing reachable is ever excluded, keep the hull as the
outer stage. Ground truth is a bounded Dijkstra over the street graph in
Phase 3. The role of this module is to cut the candidate set by a large
factor before you pay for routing.

Recommended:  hull (conservative, never drops a true positive)
           -> alpha shape (tight, cheap)
           -> network routing (truth, expensive)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .delaunay import Delaunay
from .hull import convex_hull
from .primitives import Point, circumradius, dist, ensure_ccw, polygon_area, signed_area

__all__ = ["alpha_shape", "suggest_alpha", "reachable_area"]

Edge = Tuple[int, int]


def _edge_key(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def suggest_alpha(points: Sequence[Point], percentile: float = 95.0) -> float:
    """A data-driven alpha radius: a high percentile of Delaunay edge lengths.

    Reasoning: most Delaunay edges connect genuinely nearby points, and the
    long tail is precisely the edges that leap across the empty regions we
    want to carve out. Cutting at a high percentile of edge length keeps the
    fabric and drops the leaps.

    This is a heuristic, not a theorem. Tune it against your data and look at
    the output on a map -- alpha is a shape parameter and shape parameters
    are chosen by looking. Too small and the polygon shatters into fragments
    or grows spurious holes; too large and it relaxes back toward the convex
    hull.
    """
    if len(points) < 3:
        return math.inf
    d = Delaunay(points)
    lengths = sorted(dist(d.points[i], d.points[j]) for i, j in d.edges())
    if not lengths:
        return math.inf
    k = min(len(lengths) - 1, int(len(lengths) * percentile / 100.0))
    # Circumradius scales roughly with edge length; the factor converts an
    # edge-length cut into a circumradius cut. sqrt(3)/3 is the ratio for an
    # equilateral triangle and is the natural reference.
    return lengths[k] / math.sqrt(3.0) * 2.0


def alpha_shape(
    points: Sequence[Point],
    alpha_radius: Optional[float] = None,
    percentile: float = 95.0,
) -> List[List[Point]]:
    """Concave hull. Returns a list of rings, largest first.

    Parameters
    ----------
    alpha_radius : float, optional
        Maximum circumradius, metres. Triangles above it are discarded.
        Defaults to `suggest_alpha(points, percentile)`.

    Returns
    -------
    list of rings, each a list of (x, y), no repeated closing vertex, wound
    counter-clockwise, sorted by area descending.

    Multiple rings are returned when the alpha filter disconnects the point
    set -- which is *information*, not a failure. Two rings means two
    genuinely separate reachable islands (think: this side of the tracks and
    the bit past the level crossing). Callers that need a single polygon
    should take `rings[0]`, but should probably first ask themselves why
    there was more than one.

    Holes are not distinguished from outer rings here. If your data can
    produce a reachable region with a genuine hole in it (a park you cannot
    enter, a superblock), test ring orientation and containment before
    treating rings[1:] as separate islands.
    """
    pts = list(points)
    if len(pts) < 3:
        return [list(pts)] if pts else []

    d = Delaunay(pts)
    if not d.triangles:
        return [convex_hull(pts)]

    if alpha_radius is None:
        alpha_radius = suggest_alpha(pts, percentile)

    keep: List[Tuple[int, int, int]] = []
    for t in d.triangles:
        r = circumradius(d.points[t[0]], d.points[t[1]], d.points[t[2]])
        if r <= alpha_radius:
            keep.append(t)

    if not keep:
        # Alpha too aggressive -- nothing survived. Returning an empty shape
        # here would be a silent, confusing failure, so fall back to the hull
        # and let the caller notice the shape is loose rather than absent.
        return [convex_hull(pts)]

    # Boundary of the kept region: edges owned by exactly one kept triangle.
    counts: Dict[Edge, int] = defaultdict(int)
    for i, j, k in keep:
        counts[_edge_key(i, j)] += 1
        counts[_edge_key(j, k)] += 1
        counts[_edge_key(k, i)] += 1
    boundary: Set[Edge] = {e for e, c in counts.items() if c == 1}
    if not boundary:
        return [convex_hull(pts)]

    rings_idx = _stitch_rings(boundary)
    rings = [[d.points[i] for i in r] for r in rings_idx if len(r) >= 3]
    if not rings:
        return [convex_hull(pts)]

    rings = [ensure_ccw(r) for r in rings]
    rings.sort(key=polygon_area, reverse=True)
    return rings


def _stitch_rings(boundary: Set[Edge]) -> List[List[int]]:
    """Chain a bag of undirected boundary edges into closed rings.

    Each vertex normally has degree 2 in the boundary, so the walk is
    unambiguous. Degree > 2 happens at a "pinch" where the alpha region
    touches itself at a single vertex; we just take whichever unused edge
    comes first, which splits the pinch into separate rings. That is a
    defensible reading of an ambiguous shape.
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)

    unused: Set[Edge] = set(boundary)
    rings: List[List[int]] = []

    while unused:
        a, b = next(iter(unused))
        unused.discard(_edge_key(a, b))
        ring = [a, b]
        cur = b
        while True:
            nxt = None
            for cand in adj[cur]:
                e = _edge_key(cur, cand)
                if e in unused:
                    nxt = cand
                    break
            if nxt is None:
                break  # open chain -- shouldn't happen, but don't hang on it
            unused.discard(_edge_key(cur, nxt))
            if nxt == ring[0]:
                break  # closed
            ring.append(nxt)
            cur = nxt
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def reachable_area(
    points: Sequence[Point],
    method: str = "alpha",
    alpha_radius: Optional[float] = None,
) -> List[Point]:
    """Single reachable-area ring from sampled reachable points.

    method="convex" : convex hull. Fast, conservative (never excludes a truly
                      reachable point), and wrong in the ways described at the
                      top of this file. Fine as an outer prefilter.
    method="alpha"  : alpha shape, largest ring. Tighter and shaped like a
                      real isochrone. Default.
    """
    if method == "convex":
        return convex_hull(points)
    if method == "alpha":
        rings = alpha_shape(points, alpha_radius=alpha_radius)
        return rings[0] if rings else []
    raise ValueError(f"unknown method {method!r}; expected 'convex' or 'alpha'")
