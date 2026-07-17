"""
Convex polygon clipping (Sutherland-Hodgman).

Small module, but it is what makes the Voronoi construction tractable: a
Voronoi cell is the intersection of half-planes, and intersecting a convex
polygon with a half-plane is exactly this algorithm. It also solves the
unbounded-cell problem for free -- start from the bounding box and every cell
comes out bounded, with no special case for the sites on the convex hull.

Sutherland-Hodgman is only correct when the *clip* region is convex. A
half-plane is convex and a rectangle is convex, so both uses here are valid.
It is not a general polygon-intersection routine; do not reach for it to
clip against an arbitrary isochrone (that needs Greiner-Hormann or Vatti).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .primitives import Point

__all__ = ["clip_halfplane", "clip_rect", "bisector_halfplane"]


def clip_halfplane(
    poly: Sequence[Point], a: float, b: float, c: float, eps: float = 1e-9
) -> List[Point]:
    """Clip a convex ring to the half-plane {(x, y) : a*x + b*y <= c}.

    Walk the edges; for each, emit the intersection point when the edge
    crosses the boundary line, and emit the endpoint when it is inside. The
    result is the clipped convex ring, possibly empty.
    """
    if not poly:
        return []
    out: List[Point] = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        # Signed distance to the boundary line (up to a positive scale).
        dc = a * cur[0] + b * cur[1] - c
        dn = a * nxt[0] + b * nxt[1] - c
        cur_in = dc <= eps
        nxt_in = dn <= eps

        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            denom = dc - dn
            if abs(denom) > 1e-18:
                t = dc / denom
                out.append(
                    (cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1]))
                )
    return out


def clip_rect(
    poly: Sequence[Point], minx: float, miny: float, maxx: float, maxy: float
) -> List[Point]:
    """Clip a convex ring to an axis-aligned rectangle."""
    out = list(poly)
    for a, b, c in (
        (-1.0, 0.0, -minx),  # x >= minx
        (1.0, 0.0, maxx),  # x <= maxx
        (0.0, -1.0, -miny),  # y >= miny
        (0.0, 1.0, maxy),  # y <= maxy
    ):
        out = clip_halfplane(out, a, b, c)
        if not out:
            return []
    return out


def bisector_halfplane(pi: Point, pj: Point) -> Tuple[float, float, float]:
    """Half-plane of points at least as close to `pi` as to `pj`.

    Derivation -- the algebra is worth seeing because the naive route
    (construct the perpendicular bisector line geometrically) needs a special
    case for vertical lines, and this one has none:

        |p - pi|^2 <= |p - pj|^2
        -2 p.pi + |pi|^2 <= -2 p.pj + |pj|^2
        2 (pj - pi) . p <= |pj|^2 - |pi|^2

    which is a*x + b*y <= c with the coefficients below. Every Voronoi cell
    is the intersection of these over the site's Delaunay neighbours.
    """
    xi, yi = pi
    xj, yj = pj
    a = 2.0 * (xj - xi)
    b = 2.0 * (yj - yi)
    c = (xj * xj + yj * yj) - (xi * xi + yi * yi)
    return (a, b, c)
