"""
Delaunay triangulation, incremental (Bowyer-Watson).

Why this is here even though you only asked for Voronoi
------------------------------------------------------
The Voronoi diagram and the Delaunay triangulation are duals: two sites share
a Voronoi edge exactly when they share a Delaunay edge. So Delaunay is not a
detour on the way to Voronoi, it *is* Voronoi, in the representation that is
actually easy to compute.

And having paid for it once, the alpha shape (concave.py) -- the honest
non-convex reachable-area polygon that the convex hull cannot give you --
falls out as a filter over the same triangles, for about thirty extra lines.
One data structure, two Phase 2 deliverables, plus a better answer to a third.

The algorithm
-------------
1. Start with one "super triangle" large enough to contain every input point.
2. Insert points one at a time. For each point p:
   a. Find all triangles whose *circumcircle* contains p ("bad" triangles).
      By the Delaunay property they are exactly the triangles that must die.
   b. Their union is a star-shaped cavity around p. Collect its boundary --
      the edges belonging to exactly one bad triangle.
   c. Delete the bad triangles, and fan p to every boundary edge.
3. Delete every triangle still touching a super-triangle vertex.

Two implementation choices that matter for complexity
-----------------------------------------------------
The naive version scans all O(n) triangles at step 2a for every insertion,
giving O(n^2) overall. That is the version in most tutorials and it is why
people believe Bowyer-Watson is slow. Two fixes, both here:

  * **Point location by walking.** Maintain edge->triangle adjacency and walk
    from the previously-inserted triangle toward p, crossing one edge at a
    time. With spatially-sorted insertion the previous triangle is nearby, so
    the walk is short.

  * **Cavity discovery by flood fill.** The bad triangles are connected (the
    cavity is star-shaped about p), so once located, we flood outward through
    adjacency and stop at the first good triangle in each direction. We touch
    O(size of cavity) triangles instead of O(all of them), and the cavity is
    O(1) on average.

Together these give roughly O(n log n) in practice. The worst case is still
O(n^2) for adversarial input; a proper guarantee needs randomised incremental
with a conflict graph, or Fortune's sweepline. That is the right trade to
name in an interview and the wrong one to actually build here.

Scale note: this is pure Python, and it is meant for the *sites* of a Voronoi
diagram -- the few hundred to few thousand cafés in a district -- not for all
120k POIs at once. See the benchmark in bench_phase2.py for measured numbers.
For 120k sites, reach for scipy.spatial.Delaunay (Qhull, C) and keep this as
the from-scratch artifact that proves you know what Qhull is doing.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .primitives import Point, circumcenter, cross, dedupe

Tri = Tuple[int, int, int]
Edge = Tuple[int, int]

__all__ = ["Delaunay", "Tri", "Edge"]

# How far out to place the super-triangle vertices, as a multiple of the
# input's bounding radius. Too small and triangles near the convex hull come
# out wrong; too large and the circumcircle determinant loses precision to
# the enormous coordinates. 50 is comfortably inside both failure modes for
# metre-scale input -- and test_delaunay.py verifies the Delaunay property
# exhaustively against brute force, which is what actually justifies the
# number.
_SUPER_SCALE = 50.0


def _canonical(t: Tri) -> Tri:
    """Rotate a CCW triangle so it starts at its smallest index.

    Rotation preserves orientation, so this gives a unique key per triangle
    while keeping the winding intact -- which we need, because the
    in-circumcircle predicate's sign depends on it.
    """
    i, j, k = t
    if i <= j and i <= k:
        return (i, j, k)
    if j <= i and j <= k:
        return (j, k, i)
    return (k, i, j)


def _edge_key(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


class Delaunay:
    """Delaunay triangulation of a set of 2D points.

    Attributes
    ----------
    points : list[Point]
        The de-duplicated input, in insertion-independent original order.
    triangles : list[Tri]
        Index triples into `points`, each wound counter-clockwise.
    """

    def __init__(self, points: Sequence[Point], seed: int = 0xC0FFEE) -> None:
        # Coincident points produce zero-area triangles, which have no
        # circumcentre, which breaks everything downstream. Kill them first.
        self.points: List[Point] = dedupe(points)
        n = len(self.points)
        self.triangles: List[Tri] = []
        self._adj: Dict[Edge, Set[Tri]] = {}
        self._tris: Set[Tri] = set()

        if n < 3:
            return

        self._coords: List[Point] = list(self.points)
        self._build(n, seed)

        # Publish only triangles made entirely of real points.
        self.triangles = [t for t in self._tris if max(t) < n]
        # Rebuild adjacency restricted to the published triangles, so that
        # neighbour queries never leak the super-triangle scaffolding.
        self._adj = {}
        self._tris = set(self.triangles)
        for t in self.triangles:
            self._index(t)
        del self._coords

    # -- construction -----------------------------------------------------

    def _build(self, n: int, seed: int) -> None:
        # Super triangle: an equilateral triangle circumscribing a circle that
        # comfortably contains every input point.
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        r = max(math.hypot(x - cx, y - cy) for x, y in self.points)
        r = max(r, 1.0) * _SUPER_SCALE
        s0, s1, s2 = n, n + 1, n + 2
        self._coords.extend(
            [
                (cx, cy + 2.0 * r),
                (cx - r * math.sqrt(3.0), cy - r),
                (cx + r * math.sqrt(3.0), cy - r),
            ]
        )
        # Wind it CCW.
        super_tri: Tri = (s0, s1, s2)
        if cross(self._coords[s0], self._coords[s1], self._coords[s2]) < 0:
            super_tri = (s0, s2, s1)
        self._add(super_tri)

        # Insert in spatially-sorted order, lightly shuffled. Pure spatial
        # order makes the walk short but degrades the incremental structure;
        # pure random makes the structure good but the walk long. Shuffling
        # within buckets of a spatial sort (a "BRIO"-flavoured order) gets
        # most of both.
        order = self._insertion_order(n, seed)

        hint: Tri = super_tri
        for pi in order:
            hint = self._insert(pi, hint)

    def _insertion_order(self, n: int, seed: int) -> List[int]:
        rng = random.Random(seed)
        idx = list(range(n))
        # Sort into a grid of columns, alternating direction (boustrophedon)
        # so consecutive points stay spatially close across column breaks.
        cols = max(1, int(math.sqrt(n) / 2))
        xs = [self.points[i][0] for i in idx]
        minx, maxx = min(xs), max(xs)
        span = (maxx - minx) or 1.0

        def col_of(i: int) -> int:
            return min(cols - 1, int((self.points[i][0] - minx) / span * cols))

        idx.sort(key=lambda i: (col_of(i), self.points[i][1] * (1 if col_of(i) % 2 == 0 else -1)))
        # Shuffle within small blocks to break adversarial orderings without
        # destroying locality.
        block = 8
        for s in range(0, n, block):
            chunk = idx[s : s + block]
            rng.shuffle(chunk)
            idx[s : s + block] = chunk
        return idx

    # -- mesh bookkeeping -------------------------------------------------

    def _index(self, t: Tri) -> None:
        i, j, k = t
        for a, b in ((i, j), (j, k), (k, i)):
            self._adj.setdefault(_edge_key(a, b), set()).add(t)

    def _add(self, t: Tri) -> None:
        t = _canonical(t)
        self._tris.add(t)
        self._index(t)

    def _remove(self, t: Tri) -> None:
        self._tris.discard(t)
        i, j, k = t
        for a, b in ((i, j), (j, k), (k, i)):
            e = _edge_key(a, b)
            s = self._adj.get(e)
            if s is not None:
                s.discard(t)
                if not s:
                    del self._adj[e]

    def _neighbour(self, t: Tri, a: int, b: int) -> Optional[Tri]:
        s = self._adj.get(_edge_key(a, b))
        if not s:
            return None
        for other in s:
            if other != t:
                return other
        return None

    # -- predicates -------------------------------------------------------

    def _in_circumcircle(self, t: Tri, p: Point) -> bool:
        """True if p lies strictly inside the circumcircle of CCW triangle t.

        The standard 3x3 determinant form. Translating each vertex by -p
        first is not cosmetic: it keeps the squared terms small, which is
        what stops the determinant from being swamped when the coordinates
        are large relative to the triangle.
        """
        ax, ay = self._coords[t[0]]
        bx, by = self._coords[t[1]]
        cx, cy = self._coords[t[2]]
        px, py = p
        ax -= px
        ay -= py
        bx -= px
        by -= py
        cx -= px
        cy -= py
        a2 = ax * ax + ay * ay
        b2 = bx * bx + by * by
        c2 = cx * cx + cy * cy
        det = (
            ax * (by * c2 - b2 * cy)
            - ay * (bx * c2 - b2 * cx)
            + a2 * (bx * cy - by * cx)
        )
        return det > 0.0

    def _contains(self, t: Tri, p: Point) -> bool:
        i, j, k = t
        c = self._coords
        return (
            cross(c[i], c[j], p) >= 0
            and cross(c[j], c[k], p) >= 0
            and cross(c[k], c[i], p) >= 0
        )

    # -- point location ---------------------------------------------------

    def _locate(self, p: Point, hint: Tri) -> Tri:
        """Walk from `hint` to the triangle containing p.

        At each step, find an edge with p strictly on its outside and cross
        it. Because the super triangle contains everything, this terminates
        at a real triangle. The iteration cap plus linear-scan fallback is
        insurance against a cycle from a float-noise orientation flip -- rare,
        but a silent infinite loop is not an acceptable failure mode.
        """
        t = hint if hint in self._tris else next(iter(self._tris))
        c = self._coords
        for _ in range(len(self._tris) + 8):
            if t not in self._tris:
                t = next(iter(self._tris))
            i, j, k = t
            moved = False
            for a, b in ((i, j), (j, k), (k, i)):
                if cross(c[a], c[b], p) < 0:
                    nb = self._neighbour(t, a, b)
                    if nb is not None:
                        t = nb
                        moved = True
                        break
            if not moved:
                return t
        for t in self._tris:  # fallback: exhaustive
            if self._contains(t, p):
                return t
        return next(iter(self._tris))

    # -- insertion --------------------------------------------------------

    def _insert(self, pi: int, hint: Tri) -> Tri:
        p = self._coords[pi]
        start = self._locate(p, hint)

        # Flood-fill the cavity: bad triangles are connected, so expand only
        # through neighbours of triangles already known to be bad.
        bad: List[Tri] = []
        seen: Set[Tri] = {start}
        stack: List[Tri] = [start]
        while stack:
            t = stack.pop()
            if not self._in_circumcircle(t, p):
                continue
            bad.append(t)
            i, j, k = t
            for a, b in ((i, j), (j, k), (k, i)):
                nb = self._neighbour(t, a, b)
                if nb is not None and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)

        if not bad:
            # p sits outside every circumcircle -- can only happen on float
            # noise or an exact duplicate that survived dedupe. Skipping is
            # safe: the mesh stays a valid triangulation, minus one point.
            return start

        # Cavity boundary: edges owned by exactly one bad triangle.
        counts: Dict[Edge, int] = {}
        for t in bad:
            i, j, k = t
            for a, b in ((i, j), (j, k), (k, i)):
                e = _edge_key(a, b)
                counts[e] = counts.get(e, 0) + 1
        boundary = [e for e, cnt in counts.items() if cnt == 1]

        for t in bad:
            self._remove(t)

        last: Optional[Tri] = None
        c = self._coords
        for a, b in boundary:
            o = cross(c[a], c[b], p)
            if o == 0.0:
                continue  # degenerate: p on the edge's line, zero-area triangle
            tri: Tri = (a, b, pi) if o > 0 else (b, a, pi)
            self._add(tri)
            last = _canonical(tri)

        return last if last is not None else next(iter(self._tris))

    # -- public queries ---------------------------------------------------

    def neighbours(self, i: int) -> Set[int]:
        """Indices sharing a Delaunay edge with point i.

        These are exactly the sites whose Voronoi cells can bound cell i --
        which is what makes Voronoi construction linear rather than quadratic.
        """
        out: Set[int] = set()
        for t in self.triangles:
            if i in t:
                out.update(x for x in t if x != i)
        return out

    def neighbour_map(self) -> Dict[int, Set[int]]:
        """All adjacency in one O(triangles) pass.

        `neighbours(i)` scans every triangle, so calling it in a loop over n
        sites is O(n * triangles) = O(n^2). Voronoi needs adjacency for every
        site, so it uses this instead.
        """
        m: Dict[int, Set[int]] = {i: set() for i in range(len(self.points))}
        for i, j, k in self.triangles:
            m[i].update((j, k))
            m[j].update((i, k))
            m[k].update((i, j))
        return m

    def edges(self) -> Set[Edge]:
        out: Set[Edge] = set()
        for i, j, k in self.triangles:
            out.add(_edge_key(i, j))
            out.add(_edge_key(j, k))
            out.add(_edge_key(k, i))
        return out

    def circumcenters(self) -> List[Optional[Point]]:
        """Circumcentre per triangle, aligned with `self.triangles`.

        These are the vertices of the dual Voronoi diagram.
        """
        return [
            circumcenter(self.points[i], self.points[j], self.points[k])
            for i, j, k in self.triangles
        ]

    def convex_hull_indices(self) -> List[int]:
        """Hull of the input, read off the triangulation for free.

        A Delaunay edge is on the convex hull exactly when it belongs to only
        one triangle. Nice as a cross-check against hull.convex_hull -- two
        completely independent algorithms that must agree.
        """
        boundary: Dict[Edge, int] = {}
        for i, j, k in self.triangles:
            for a, b in ((i, j), (j, k), (k, i)):
                e = _edge_key(a, b)
                boundary[e] = boundary.get(e, 0) + 1
        return sorted({v for e, c in boundary.items() if c == 1 for v in e})

    def __len__(self) -> int:
        return len(self.triangles)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Delaunay({len(self.points)} points, {len(self.triangles)} triangles)"
