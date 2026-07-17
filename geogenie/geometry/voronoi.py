"""
Voronoi diagram -- "nearest POI catchment zones".

The definition is the whole point: cell(i) is the set of locations whose
nearest site is site i. So a Voronoi diagram over every café in a city *is*
the answer to "which café would I be sent to if I were standing here", for
every possible here, precomputed.

Construction: half-plane intersection, not dual-graph traversal
---------------------------------------------------------------
The textbook construction walks the Delaunay dual: Voronoi vertices are
triangle circumcentres, and you connect the circumcentres of triangles that
share an edge. It works, but it has an ugly wart -- cells belonging to sites
on the convex hull are **unbounded**, so the dual walk produces dangling rays
that need their own representation, their own clipping code, and their own
bugs. Roughly a third of the code exists to handle sites on the boundary.

Instead, build each cell directly from its definition:

    cell(i) = intersection over j != i of { p : |p - pi| <= |p - pj| }

Each of those is a half-plane (clip.bisector_halfplane). Start from the
bounding box and clip. Unbounded cells are bounded by the box automatically,
with no special case, no rays, and no branch. Every cell is a plain convex
polygon.

The cost of the naive form is that intersecting over *all* j is O(n) per
cell, so O(n^2) overall. This is where Delaunay earns its keep: two Voronoi
cells are adjacent only if their sites share a Delaunay edge, so clipping
against just the Delaunay neighbours is sufficient and gives the identical
answer. Average Delaunay degree is under 6 regardless of n, so the whole
diagram is O(n) clips after the triangulation.

`brute_force=True` clips against all sites instead, which is O(n^2) and
exists so the fast path can be tested against a definitionally-obvious
reference.

Cross-validation with Phase 1
-----------------------------
There is a free, very strong correctness test available here, and
tests/test_voronoi.py uses it: for any query point q,

    the Voronoi cell containing q  ==  the site your k-d tree returns as
                                       nearest neighbour of q

Two entirely independent implementations -- a Phase 1 tree search and a
Phase 2 computational-geometry construction -- that must agree on every
input. If they ever disagree, one of them is broken and you have a
reproducing case for free.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .clip import bisector_halfplane, clip_halfplane
from .delaunay import Delaunay
from .point_in_polygon import PreparedPolygon
from .primitives import Point, bbox, dist2, polygon_area, polygon_centroid

__all__ = ["VoronoiDiagram"]


class VoronoiDiagram:
    """Bounded Voronoi cells for a set of sites.

    Parameters
    ----------
    sites : sequence of (x, y) in metres
    bounds : (minx, miny, maxx, maxy), optional
        The clipping window. Defaults to the sites' bounding box expanded by
        `margin`. Cells are only meaningful inside this window -- the
        boundary of an unbounded cell is an artifact of the window, not
        geometry, which is why `is_unbounded` exists to flag them.
    margin : float
        Expansion of the default window, in metres.
    brute_force : bool
        Clip against all sites instead of Delaunay neighbours. O(n^2).
        Reference implementation for tests.
    """

    def __init__(
        self,
        sites: Sequence[Point],
        bounds: Optional[Tuple[float, float, float, float]] = None,
        margin: float = 500.0,
        brute_force: bool = False,
    ) -> None:
        self.sites: List[Point] = [tuple(s) for s in sites]
        n = len(self.sites)
        if n == 0:
            raise ValueError("VoronoiDiagram needs at least one site")

        if bounds is None:
            minx, miny, maxx, maxy = bbox(self.sites)
            # A degenerate (zero-width) bbox would clip every cell to nothing.
            if maxx - minx < 1e-9:
                minx, maxx = minx - 1.0, maxx + 1.0
            if maxy - miny < 1e-9:
                miny, maxy = miny - 1.0, maxy + 1.0
            bounds = (minx - margin, miny - margin, maxx + margin, maxy + margin)
        self.bounds = bounds

        self.delaunay: Optional[Delaunay] = None
        self._neighbours: Dict[int, Iterable[int]]

        if brute_force or n < 3:
            self._neighbours = {i: [j for j in range(n) if j != i] for i in range(n)}
        else:
            self.delaunay = Delaunay(self.sites)
            # Delaunay de-duplicates; if that changed the point set the index
            # mapping would be wrong, so fail loudly rather than misalign.
            if len(self.delaunay.points) != n:
                raise ValueError(
                    f"sites contain duplicates ({n} in, {len(self.delaunay.points)} unique); "
                    "de-duplicate before constructing a Voronoi diagram -- coincident "
                    "sites have no well-defined cells"
                )
            nmap = self.delaunay.neighbour_map()
            # An isolated site (no Delaunay edges) can only arise from a
            # degenerate triangulation. Fall back to all-pairs for it.
            self._neighbours = {
                i: (nmap[i] if nmap[i] else [j for j in range(n) if j != i])
                for i in range(n)
            }

        self.cells: List[List[Point]] = [self._build_cell(i) for i in range(n)]
        self._prepared: List[Optional[PreparedPolygon]] = [None] * n

    # -- construction -----------------------------------------------------

    def _build_cell(self, i: int) -> List[Point]:
        minx, miny, maxx, maxy = self.bounds
        poly: List[Point] = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        pi = self.sites[i]
        for j in self._neighbours[i]:
            if j == i:
                continue
            a, b, c = bisector_halfplane(pi, self.sites[j])
            poly = clip_halfplane(poly, a, b, c)
            if not poly:
                break
        return poly

    # -- queries ----------------------------------------------------------

    def cell(self, i: int) -> List[Point]:
        return self.cells[i]

    def cell_area(self, i: int) -> float:
        """Area of the catchment zone, m^2.

        Directly interpretable for GeoGenie: a café with a large cell is the
        nearest option over a wide area -- it is "underserved territory", and
        a plausible ranking signal in its own right. A tiny cell means the
        site is one of a dense cluster.
        """
        return polygon_area(self.cells[i]) if len(self.cells[i]) >= 3 else 0.0

    def cell_centroid(self, i: int) -> Point:
        return polygon_centroid(self.cells[i])

    def is_unbounded(self, i: int, eps: float = 1e-6) -> bool:
        """True if the cell touches the clipping window.

        Such a cell is genuinely infinite in the mathematical diagram; what
        you see is a window artifact. Its area is meaningless and must not be
        used as a ranking signal -- which is exactly the sort of thing that
        silently poisons a model if nobody flags it.
        """
        minx, miny, maxx, maxy = self.bounds
        for x, y in self.cells[i]:
            if (
                abs(x - minx) < eps
                or abs(x - maxx) < eps
                or abs(y - miny) < eps
                or abs(y - maxy) < eps
            ):
                return True
        return False

    def locate(self, p: Point) -> int:
        """Index of the cell containing p -- i.e. the nearest site.

        O(n) linear scan over prepared cells (each with a bbox reject). This
        is deliberately the *slow* way to answer a nearest-neighbour query
        and it is here for validation, not for production. If you want the
        nearest site, use the Phase 1 k-d tree: it is O(log n) and it does
        not require building a diagram first.

        The value of this method is that it is a completely independent
        second opinion, so it can prove the k-d tree right.
        """
        for i in range(len(self.sites)):
            if len(self.cells[i]) < 3:
                continue
            pp = self._prepared[i]
            if pp is None:
                pp = PreparedPolygon(self.cells[i])
                self._prepared[i] = pp
            if pp.contains(p):
                return i
        # p is outside the clipping window, or landed exactly on a boundary
        # that float noise pushed out of every cell. Fall back to the
        # definition itself.
        return min(range(len(self.sites)), key=lambda i: dist2(p, self.sites[i]))

    def neighbours(self, i: int) -> List[int]:
        """Sites whose cells share an edge with cell i -- the Delaunay
        neighbours. Useful for "show me the next-nearest alternatives"."""
        return sorted(self._neighbours[i])

    def to_geojson(self, ltp=None, properties: Optional[Sequence[dict]] = None) -> dict:
        """FeatureCollection of the cells, for dropping straight into
        geojson.io / Leaflet / kepler.gl to eyeball the result.

        Pass the `LocalTangentPlane` you projected with to emit real lat/lon;
        without it the coordinates stay in metres, which no map will read.
        """
        feats = []
        for i, cell in enumerate(self.cells):
            if len(cell) < 3:
                continue
            ring = list(cell)
            if ltp is not None:
                ring = [(lon, lat) for lat, lon in ltp.to_latlon_many(ring)]
            ring.append(ring[0])  # GeoJSON rings must close
            props = {"site_index": i, "area_m2": round(self.cell_area(i), 2),
                     "unbounded": self.is_unbounded(i)}
            if properties is not None and i < len(properties):
                props.update(properties[i])
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": props,
                }
            )
        return {"type": "FeatureCollection", "features": feats}

    def __len__(self) -> int:
        return len(self.sites)

    def __repr__(self) -> str:  # pragma: no cover
        return f"VoronoiDiagram({len(self.sites)} sites)"
