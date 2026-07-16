"""
Voronoi diagram -- "which POI is nearest here?" catchment zones.

THE DEFINITION
--------------
The Voronoi cell of site i is the set of all points closer to site i than to
any other site. Partition the plane by nearest site and you get catchment
zones: "everyone standing in this polygon should be sent to that cafe."

That definition is also, read literally, an algorithm:

    cell(i) = intersection over all j != i of
              { p : |p - site_i| <= |p - site_j| }

Each of those sets is a HALF-PLANE bounded by the perpendicular bisector of
sites i and j. So a Voronoi cell is just an intersection of half-planes, and
intersecting half-planes is something we can do by repeated polygon clipping.

That is exactly what this file does. Start with a big rectangle, clip it by
the bisector against every other site, and whatever survives is the cell.

WHY THIS ALGORITHM, AND NOT FORTUNE'S
--------------------------------------
This is O(n^2): n sites, each clipped against n-1 bisectors. Fortune's sweep
line is O(n log n), and Bowyer-Watson Delaunay + dualisation is O(n log n)
expected. Both are strictly better asymptotically.

They are also both substantially harder to get right -- Fortune's needs a
beach line, circle events, and careful degeneracy handling; the Delaunay dual
needs circumcentres, an unbounded-cell strategy, and index bookkeeping.

This version is ~40 lines, and its correctness is visible by inspection: it
is a direct transcription of the definition. For the actual use case -- a few
hundred POIs in a map viewport -- O(n^2) on 300 sites is ~90k clips, which
runs in milliseconds. The asymptotics do not bite until n is in the thousands.

THE UPGRADE PATH, when you need it: a cell only ever touches its DELAUNAY
NEIGHBOURS, of which there are ~6 on average regardless of n. So once you
have a Delaunay triangulation, you clip each cell against ~6 bisectors
instead of n-1, and the whole thing drops to O(n log n). The clipping code
below does not change at all -- only the loop over `range(n)` narrows to a
neighbour list. Build it when a profiler tells you to, not before.

SCALE WARNING
-------------
Do not call this on all 120k POIs. It is O(n^2) and the result is a map no
human can read. Use your k-d tree to pull the POIs in the current viewport,
then build the diagram on those.

UNBOUNDED CELLS
---------------
Cells on the convex hull of the site set are infinite -- they extend forever
away from the other sites. Infinity does not fit in a polygon, so every cell
is clipped to a finite `bounds` rectangle.

Be honest about what that means: for a hull site, part of its cell boundary
is the RECTANGLE, not real geometry. Zoom out and that edge moves. Those
cells are flagged by `is_unbounded` so you can dim them, drop them, or pad
the query rectangle beyond the viewport and let the artifacts fall outside
the visible area. That last one is what mapping tools actually do.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .primitives import Point, bbox, dedupe, polygon_area

__all__ = ["VoronoiDiagram", "clip_halfplane", "bisector"]


def bisector(pi: Point, pj: Point) -> Tuple[float, float, float]:
    """Perpendicular bisector of pi and pj, as a half-plane (a, b, c).

    Returns coefficients such that the half-plane

        a*x + b*y + c <= 0

    is exactly the set of points at least as close to `pi` as to `pj`.

    Derivation (worth following once -- it is three lines of algebra):

        |p - pi|^2 <= |p - pj|^2
        x^2 - 2*ax*x + ax^2 + y^2 - 2*ay*y + ay^2
            <= x^2 - 2*bx*x + bx^2 + y^2 - 2*by*y + by^2

    The x^2 and y^2 terms cancel -- which is *why* the bisector is a straight
    line and not a conic -- leaving

        2*(bx-ax)*x + 2*(by-ay)*y + (ax^2+ay^2 - bx^2-by^2) <= 0
    """
    ax, ay = pi
    bx, by = pj
    a = 2.0 * (bx - ax)
    b = 2.0 * (by - ay)
    c = (ax * ax + ay * ay) - (bx * bx + by * by)
    return (a, b, c)


def clip_halfplane(
    poly: Sequence[Point], a: float, b: float, c: float
) -> List[Point]:
    """Clip a convex polygon by the half-plane a*x + b*y + c <= 0.

    This is one pass of the Sutherland-Hodgman algorithm. Walk each edge of
    the ring and decide what to emit:

        current inside, next inside   -> keep current
        current inside, next outside  -> keep current + the crossing point
        current outside, next inside  -> keep the crossing point
        current outside, next outside -> emit nothing

    An intersection of half-planes is always convex, and Sutherland-Hodgman
    is exact for convex subject polygons, so this composes safely: clip by
    one bisector, then another, and the result stays a valid convex cell.
    (Sutherland-Hodgman's notorious degenerate-connector artifacts only
    afflict CONCAVE subject polygons, which cannot arise here.)

    Returns [] if the polygon is entirely clipped away -- which is a real
    outcome, not an error: a site enclosed by others can have an empty cell
    within the given bounds.
    """
    n = len(poly)
    if n == 0:
        return []

    out: List[Point] = []
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        f_cur = a * cur[0] + b * cur[1] + c
        f_nxt = a * nxt[0] + b * nxt[1] + c

        if f_cur <= 0.0:
            out.append(cur)

        # Strict sign change => the edge crosses the boundary line.
        # Using strict < and > means an endpoint exactly ON the line is not
        # duplicated as a crossing point (it was already emitted above).
        if (f_cur < 0.0 < f_nxt) or (f_nxt < 0.0 < f_cur):
            t = f_cur / (f_cur - f_nxt)  # safe: signs differ, so f_cur != f_nxt
            out.append(
                (cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1]))
            )

    return out


class VoronoiDiagram:
    """Bounded Voronoi cells for a set of sites.

    Parameters
    ----------
    sites : (x, y) in metres. Must be distinct -- coincident sites have no
            well-defined cell, so duplicates raise rather than silently
            producing an empty polygon.
    bounds : (minx, miny, maxx, maxy) clipping window. Defaults to the sites'
             bounding box grown by `margin`.
    margin : how far past the sites the default window extends, in metres.

    Attributes
    ----------
    cells : list of rings, parallel to `sites`. cells[i] is site i's polygon,
            CCW, possibly [] if fully clipped away.

    Complexity: O(n^2) clips, each O(cell vertices). See module note.
    """

    __slots__ = ("sites", "bounds", "cells")

    def __init__(
        self,
        sites: Sequence[Point],
        bounds: Optional[Tuple[float, float, float, float]] = None,
        margin: float = 500.0,
    ) -> None:
        pts = [(float(s[0]), float(s[1])) for s in sites]
        if not pts:
            raise ValueError("VoronoiDiagram needs at least one site")
        if len(dedupe(pts)) != len(pts):
            raise ValueError(
                "sites contain duplicates; coincident sites have no "
                "well-defined Voronoi cell -- de-duplicate first"
            )
        self.sites: List[Point] = pts

        if bounds is None:
            minx, miny, maxx, maxy = bbox(pts)
            # A single site, or perfectly collinear sites, gives a
            # zero-width or zero-height box that would clip every cell to
            # nothing. Inflate it.
            if maxx - minx < 1e-9:
                minx, maxx = minx - 1.0, maxx + 1.0
            if maxy - miny < 1e-9:
                miny, maxy = miny - 1.0, maxy + 1.0
            bounds = (minx - margin, miny - margin, maxx + margin, maxy + margin)
        self.bounds = bounds

        self.cells: List[List[Point]] = [self._build_cell(i) for i in range(len(pts))]

    def _build_cell(self, i: int) -> List[Point]:
        """Clip the window by the bisector against every other site."""
        minx, miny, maxx, maxy = self.bounds
        # Start from the whole window, CCW.
        poly: List[Point] = [
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
        ]

        pi = self.sites[i]
        for j, pj in enumerate(self.sites):
            if j == i:
                continue
            a, b, c = bisector(pi, pj)
            poly = clip_halfplane(poly, a, b, c)
            if not poly:
                break  # fully clipped; no cell within these bounds
        return poly

    # -- queries -----------------------------------------------------------

    def cell(self, i: int) -> List[Point]:
        return self.cells[i]

    def is_unbounded(self, i: int, tol: float = 1e-6) -> bool:
        """True if site i's cell touches the clipping window.

        Such a cell is infinite in reality; the part of its boundary that
        lies on the window is an artifact of `bounds`, not geometry. Do not
        present those edges as real catchment boundaries.
        """
        minx, miny, maxx, maxy = self.bounds
        for x, y in self.cells[i]:
            if (
                abs(x - minx) < tol
                or abs(x - maxx) < tol
                or abs(y - miny) < tol
                or abs(y - maxy) < tol
            ):
                return True
        return False

    def cell_area(self, i: int) -> float:
        """Catchment area in square metres. The demo-friendly statistic:
        'this cafe serves the largest area of anywhere in the district'."""
        return polygon_area(self.cells[i])

    def __len__(self) -> int:
        return len(self.sites)
