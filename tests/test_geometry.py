"""
Phase 2 test suite.

The guiding principle: every non-trivial algorithm is checked against an
*independent* reference, not against its own output. A test that only asserts
"the hull has 5 points" tells you nothing when the algorithm is wrong. A test
that asserts "no input point lies outside the hull, checked by an unrelated
predicate" cannot pass a broken hull.

Reference oracles used here:
  convex hull   -> every point is inside-or-on the hull (ray casting)
                -> and the two hull algorithms must agree
                -> and Delaunay's boundary must agree with both
  point-in-poly -> O(log n) convex path must match O(n) ray casting
  Douglas-P.    -> measured max deviation must respect the tolerance
  Delaunay      -> brute-force empty-circumcircle check, O(n * triangles)
  Voronoi       -> O(n) Delaunay-neighbour build must match the O(n^2)
                   all-pairs build; and cell containment must match
                   brute-force nearest-site
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from geogenie.geometry import (  # noqa: E402
    Delaunay,
    PreparedPolygon,
    VoronoiDiagram,
    alpha_shape,
    circumcenter,
    convex_hull,
    dist,
    dist2,
    douglas_peucker,
    graham_scan,
    hull_diameter,
    is_convex_ring,
    monotone_chain,
    point_in_convex_polygon,
    point_in_polygon,
    point_segment_distance,
    polygon_area,
    radial_distance_filter,
    signed_area,
    simplify,
    visvalingam_whyatt,
    winding_number,
)
# Projection only via core.coords (coord-funnel rule).
from geogenie.core.coords import Origin, haversine, to_lonlat, to_xy  # noqa: E402


class LocalTangentPlane:
    """Thin adapter so legacy geometry tests keep working without importing projection."""

    def __init__(self, lat0: float, lon0: float) -> None:
        self._origin = Origin(lon0, lat0)
        self.lat0 = lat0
        self.lon0 = lon0

    @classmethod
    def from_points(cls, latlons):
        lats = [p[0] for p in latlons]
        lons = [p[1] for p in latlons]
        return cls((min(lats) + max(lats)) / 2.0, (min(lons) + max(lons)) / 2.0)

    def to_xy(self, lat: float, lon: float):
        return to_xy(lon, lat, self._origin)

    def to_latlon(self, x: float, y: float):
        lon, lat = to_lonlat(x, y, self._origin)
        return (lat, lon)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- projection
def test_projection() -> None:
    section("projection")
    ltp = LocalTangentPlane(51.5074, -0.1278)  # London

    # Round trip must be exact to float precision.
    worst = 0.0
    rng = random.Random(1)
    for _ in range(2000):
        lat = 51.5074 + rng.uniform(-0.2, 0.2)
        lon = -0.1278 + rng.uniform(-0.3, 0.3)
        x, y = ltp.to_xy(lat, lon)
        lat2, lon2 = ltp.to_latlon(x, y)
        worst = max(worst, abs(lat - lat2), abs(lon - lon2))
    check("round trip is lossless", worst < 1e-9, f"worst={worst:g}")

    # Projected distance must track the great-circle distance.
    worst_rel = 0.0
    for _ in range(2000):
        a = (51.5074 + rng.uniform(-0.15, 0.15), -0.1278 + rng.uniform(-0.2, 0.2))
        b = (51.5074 + rng.uniform(-0.15, 0.15), -0.1278 + rng.uniform(-0.2, 0.2))
        hav = haversine(a[0], a[1], b[0], b[1])
        pa, pb = ltp.to_xy(*a), ltp.to_xy(*b)
        euc = dist(pa, pb)
        if hav > 100:
            worst_rel = max(worst_rel, abs(euc - hav) / hav)
    check(
        "distance error < 0.1% over a ~30km box",
        worst_rel < 1e-3,
        f"worst_rel={worst_rel:.2%}",
    )

    # Poles must be rejected, not silently mangled.
    try:
        LocalTangentPlane(89.9, 0.0)
        check("rejects polar origin", False)
    except ValueError:
        check("rejects polar origin", True)

    # Antimeridian must not produce a 40 000 km jump.
    ltp2 = LocalTangentPlane(0.0, 179.9)
    x, _ = ltp2.to_xy(0.0, -179.9)
    check("handles antimeridian wrap", abs(x) < 30_000, f"x={x:.1f} m")


# ---------------------------------------------------------------------- hull
def test_hull() -> None:
    section("convex hull")
    rng = random.Random(42)

    for trial in range(30):
        pts = [(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)) for _ in range(200)]
        h = convex_hull(pts)
        g = graham_scan(pts)

        # Oracle 1: every input point is inside or on the hull.
        outside = [p for p in pts if not point_in_polygon(p, h, include_boundary=True)]
        if outside:
            check("all points within hull", False, f"trial {trial}: {len(outside)} outside")
            return

        # Oracle 2: the hull is actually convex and CCW.
        if not is_convex_ring(h) or signed_area(h) <= 0:
            check("hull is convex + CCW", False, f"trial {trial}")
            return

        # Oracle 3: two independent algorithms must agree as a set.
        if set(h) != set(g):
            check("monotone chain == graham scan", False, f"trial {trial}")
            return

    check("all points within hull (30 random trials)", True)
    check("hull is convex + CCW (30 trials)", True)
    check("monotone chain == graham scan (30 trials)", True)

    # Degenerate inputs must not crash or lie.
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    h = convex_hull(sq + [(5, 5), (2, 3)])  # interior points must be dropped
    check("interior points excluded", len(h) == 4 and set(h) == set(sq))

    collinear = [(float(i), float(i)) for i in range(10)]
    h = convex_hull(collinear)
    check("collinear input degrades gracefully", len(h) == 2, f"got {h}")

    dupes = [(1.0, 1.0)] * 50
    h = convex_hull(dupes)
    check("all-duplicate input", len(h) == 1, f"got {h}")

    check("empty input", convex_hull([]) == [])
    check("single point", convex_hull([(3.0, 4.0)]) == [(3.0, 4.0)])

    # Collinear points on a hull edge: dropped by default, kept on request.
    edge = [(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)]
    check("collinear edge point dropped by default", len(monotone_chain(edge)) == 4)
    check(
        "collinear edge point kept on request",
        len(monotone_chain(edge, include_collinear=True)) == 5,
    )

    # Rotating calipers vs brute-force farthest pair.
    for _ in range(10):
        pts = [(rng.uniform(-500, 500), rng.uniform(-500, 500)) for _ in range(60)]
        h = convex_hull(pts)
        d_cal, _, _ = hull_diameter(h)
        d_brute = max(dist(a, b) for a in pts for b in pts)
        if abs(d_cal - d_brute) > 1e-6:
            check("rotating calipers == brute force", False, f"{d_cal} vs {d_brute}")
            return
    check("rotating calipers == brute force diameter", True)


# ------------------------------------------------------------ point in polygon
def test_pip() -> None:
    section("point in polygon")
    sq = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    check("centre inside", point_in_polygon((5, 5), sq))
    check("far outside", not point_in_polygon((50, 5), sq))
    check("outside below", not point_in_polygon((5, -1), sq))
    check("vertex counts as boundary", point_in_polygon((0, 0), sq, include_boundary=True))
    check("vertex excluded when asked", not point_in_polygon((0, 0), sq, include_boundary=False))
    check("edge midpoint on boundary", point_in_polygon((5, 0), sq, include_boundary=True))

    # The classic ray-casting killer: a ray that passes exactly through
    # vertices. Every one of these y values hits a vertex dead on.
    spike = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    check("ray through vertex: inside point", point_in_polygon((5.0, 7.0), spike))
    check("ray through vertex: notch is outside", not point_in_polygon((5.0, 1.0), spike))
    check("ray at vertex height y=5", point_in_polygon((5.0, 5.5), spike))

    # Concave polygon, checked against the winding number -- a different
    # algorithm that must agree everywhere except on the boundary.
    rng = random.Random(7)
    disagree = 0
    for _ in range(5000):
        p = (rng.uniform(-1, 11), rng.uniform(-1, 11))
        rc = point_in_polygon(p, spike, include_boundary=False)
        wn = winding_number(p, spike) != 0
        if rc != wn:
            disagree += 1
    check("ray casting == winding number on concave ring", disagree == 0, f"{disagree} disagree")

    # O(log n) convex path must match O(n) ray casting exactly.
    pts = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(80)]
    hull = convex_hull(pts)
    disagree = 0
    for _ in range(5000):
        p = (rng.uniform(-120, 120), rng.uniform(-120, 120))
        if point_in_convex_polygon(p, hull) != point_in_polygon(p, hull):
            disagree += 1
    check("O(log n) convex == O(n) ray cast", disagree == 0, f"{disagree} disagree")

    # PreparedPolygon must not change any answer.
    pp = PreparedPolygon(hull)
    check("prepared polygon detects convexity", pp.convex)
    disagree = sum(
        1
        for _ in range(3000)
        for p in [(rng.uniform(-120, 120), rng.uniform(-120, 120))]
        if pp.contains(p) != point_in_polygon(p, hull)
    )
    check("PreparedPolygon == reference", disagree == 0, f"{disagree} disagree")

    ppc = PreparedPolygon(spike)
    check("prepared polygon detects concavity", not ppc.convex)

    check("closing vertex accepted", len(PreparedPolygon(sq + [sq[0]]).ring) == 4)
    try:
        PreparedPolygon([(0, 0), (1, 1)])
        check("rejects degenerate ring", False)
    except ValueError:
        check("rejects degenerate ring", True)


# ------------------------------------------------------------------ simplify
def test_simplify() -> None:
    section("douglas-peucker")

    line = [(0.0, 0.0), (1.0, 0.1), (2.0, -0.1), (3.0, 0.05), (4.0, 0.0)]
    s = douglas_peucker(line, 0.5)
    check("near-straight line collapses to endpoints", s == [(0.0, 0.0), (4.0, 0.0)], f"{s}")

    check("tolerance 0 keeps everything", len(douglas_peucker(line, 0.0)) == len(line))
    check("2 points unchanged", douglas_peucker([(0, 0), (1, 1)], 5.0) == [(0, 0), (1, 1)])
    check("1 point unchanged", douglas_peucker([(0, 0)], 5.0) == [(0, 0)])
    check("empty unchanged", douglas_peucker([], 5.0) == [])

    # A genuine spike must survive -- this is the property that distinguishes
    # DP from a naive smoother.
    spike = [(0.0, 0.0), (1.0, 0.0), (2.0, 10.0), (3.0, 0.0), (4.0, 0.0)]
    s = douglas_peucker(spike, 1.0)
    check("spike is preserved", (2.0, 10.0) in s, f"{s}")

    # The core guarantee: no original point deviates from the simplified
    # polyline by more than the tolerance. Measured, not assumed.
    rng = random.Random(11)
    track = []
    x, y = 0.0, 0.0
    for i in range(3000):
        x += rng.uniform(0.5, 2.0)
        y += rng.uniform(-1.5, 1.5)
        track.append((x, y))

    for tol in (0.5, 2.0, 10.0, 50.0):
        s = douglas_peucker(track, tol)
        worst = 0.0
        for p in track:
            d = min(
                point_segment_distance(p, s[i], s[i + 1]) for i in range(len(s) - 1)
            )
            worst = max(worst, d)
        ok = worst <= tol + 1e-9
        check(
            f"max deviation <= tolerance (tol={tol})",
            ok,
            f"worst={worst:.4f}",
        )
        check(f"endpoints preserved (tol={tol})", s[0] == track[0] and s[-1] == track[-1])

    s = douglas_peucker(track, 10.0)
    check(
        f"compression is real ({len(track)} -> {len(s)})",
        len(s) < len(track) * 0.2,
    )

    # No recursion limit: a 60k-point trace would blow the stack in the
    # textbook recursive formulation.
    big = [(float(i), math.sin(i / 50.0) * 3.0) for i in range(60_000)]
    s = douglas_peucker(big, 0.05)
    check(f"60k points, no stack overflow ({len(s)} kept)", 2 < len(s) < 60_000)

    # Radial prefilter must not break the DP guarantee.
    dense = []
    for p in track[:500]:
        dense.extend([p] * 3)  # simulate a stationary GPS receiver
    r = radial_distance_filter(dense, 0.5)
    check(f"radial filter strips stationary fixes ({len(dense)} -> {len(r)})", len(r) < len(dense) / 2)

    s_fast = simplify(track, 10.0)
    s_hq = simplify(track, 10.0, high_quality=True)
    check("simplify() wrapper runs both paths", len(s_fast) > 2 and len(s_hq) > 2)

    # Visvalingam hits an exact budget, which is its whole reason to exist.
    v = visvalingam_whyatt(track[:300], 50)
    check("visvalingam hits exact target", len(v) == 50, f"{len(v)}")
    check("visvalingam keeps endpoints", v[0] == track[0] and v[-1] == track[299])


# ------------------------------------------------------------------ delaunay
def _delaunay_property_holds(d: Delaunay, tol: float = 1e-6) -> tuple[bool, str]:
    """Brute force: no point may lie strictly inside any triangle's
    circumcircle. This is the *definition* of Delaunay, so checking it
    directly is the strongest test available. O(n * triangles)."""
    for t in d.triangles:
        a, b, c = d.points[t[0]], d.points[t[1]], d.points[t[2]]
        cc = circumcenter(a, b, c)
        if cc is None:
            return False, f"degenerate triangle {t}"
        r2 = dist2(cc, a)
        for i, p in enumerate(d.points):
            if i in t:
                continue
            if dist2(p, cc) < r2 * (1 - tol):
                return False, f"point {i} inside circumcircle of {t}"
    return True, ""


def test_delaunay() -> None:
    section("delaunay")
    rng = random.Random(99)

    for n in (10, 50, 150):
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
        d = Delaunay(pts)
        ok, why = _delaunay_property_holds(d)
        check(f"empty circumcircle property holds (n={n})", ok, why)

        # Euler: a triangulation of n points with h on the hull has
        # 2n - 2 - h triangles. An independent count that must line up.
        h = len(convex_hull(pts))
        expected = 2 * n - 2 - h
        check(
            f"triangle count == 2n-2-h (n={n})",
            len(d.triangles) == expected,
            f"got {len(d.triangles)}, want {expected}",
        )

        # Delaunay's hull, read off the boundary edges, must match the
        # monotone chain's hull. Two unrelated algorithms.
        dh = set(d.convex_hull_indices())
        mh = {i for i, p in enumerate(d.points) if p in set(convex_hull(pts))}
        check(f"delaunay hull == monotone chain hull (n={n})", dh == mh)

        # Every triangle must be wound CCW -- the circumcircle predicate's
        # sign depends on it, so a single CW triangle silently corrupts the
        # whole mesh.
        cw = [t for t in d.triangles if signed_area([d.points[i] for i in t]) <= 0]
        check(f"all triangles CCW (n={n})", not cw, f"{len(cw)} clockwise")

    check("fewer than 3 points -> no triangles", len(Delaunay([(0, 0), (1, 1)]).triangles) == 0)

    # Points on a perfect grid are maximally degenerate: every set of four
    # neighbours is co-circular, so the triangulation is not unique and the
    # predicate is on a knife edge. If Bowyer-Watson survives this it will
    # survive real data.
    grid = [(float(i) * 10, float(j) * 10) for i in range(9) for j in range(9)]
    d = Delaunay(grid)
    ok, why = _delaunay_property_holds(d, tol=1e-6)
    check("survives a co-circular grid", ok, why)

    # Duplicates must be removed, not crash.
    d = Delaunay([(0, 0), (10, 0), (5, 8), (0, 0), (10, 0)])
    check("duplicates de-duplicated", len(d.points) == 3)

    # Collinear input has no triangulation; it must return empty, not hang.
    d = Delaunay([(float(i), 0.0) for i in range(10)])
    check("collinear input -> no triangles", len(d.triangles) == 0)


# ------------------------------------------------------------------- voronoi
def test_voronoi() -> None:
    section("voronoi")
    rng = random.Random(5)
    sites = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(60)]

    fast = VoronoiDiagram(sites)
    slow = VoronoiDiagram(sites, bounds=fast.bounds, brute_force=True)

    # The O(n) Delaunay-neighbour build must produce identical cells to the
    # O(n^2) all-pairs build. This is the test that justifies the whole
    # "clip against Delaunay neighbours only" optimisation.
    worst = 0.0
    for i in range(len(sites)):
        a, b = fast.cell_area(i), slow.cell_area(i)
        if max(a, b) > 0:
            worst = max(worst, abs(a - b) / max(a, b))
    check("delaunay-neighbour build == all-pairs build", worst < 1e-6, f"worst rel diff {worst:g}")

    # Cells must tile the clipping window exactly -- no gaps, no overlaps.
    minx, miny, maxx, maxy = fast.bounds
    total = sum(fast.cell_area(i) for i in range(len(sites)))
    box = (maxx - minx) * (maxy - miny)
    check("cells tile the window", abs(total - box) / box < 1e-9, f"{total:.2f} vs {box:.2f}")

    # Every cell must contain its own site.
    bad = [i for i in range(len(sites)) if not PreparedPolygon(fast.cells[i]).contains(sites[i])]
    check("every cell contains its site", not bad, f"{bad}")

    # Cells must be convex -- a Voronoi cell is an intersection of
    # half-planes, so a non-convex one means the clipper is broken.
    bad = [i for i in range(len(sites)) if not is_convex_ring(fast.cells[i])]
    check("all cells convex", not bad, f"{bad}")

    # THE key test: cell membership must equal nearest-site. This is the
    # cross-check your Phase 1 k-d tree slots into -- swap the brute-force
    # nearest below for kdtree.nearest() and it validates both phases at once.
    disagree = 0
    for _ in range(3000):
        q = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        by_cell = fast.locate(q)
        by_dist = min(range(len(sites)), key=lambda i: dist2(q, sites[i]))
        if by_cell != by_dist:
            # Only a genuine tie (equidistant) is an acceptable disagreement.
            if abs(dist2(q, sites[by_cell]) - dist2(q, sites[by_dist])) > 1e-6:
                disagree += 1
    check("cell containment == nearest site (3000 probes)", disagree == 0, f"{disagree} disagree")

    # Hull sites have unbounded cells; interior sites should not.
    hull_pts = set(convex_hull(sites))
    unbounded = {i for i in range(len(sites)) if fast.is_unbounded(i)}
    hull_idx = {i for i, s in enumerate(sites) if s in hull_pts}
    check("hull sites are flagged unbounded", hull_idx <= unbounded, f"{hull_idx - unbounded}")

    # Two sites: the diagram is a single perpendicular bisector.
    v2 = VoronoiDiagram([(0.0, 0.0), (100.0, 0.0)])
    check("two sites -> two cells", len(v2.cells) == 2)
    check("two sites: left cell owns a left point", v2.locate((-10.0, 0.0)) == 0)
    check("two sites: right cell owns a right point", v2.locate((110.0, 0.0)) == 1)

    v1 = VoronoiDiagram([(5.0, 5.0)])
    check("single site owns the whole window", abs(polygon_area(v1.cells[0]) - polygon_area(
        [(v1.bounds[0], v1.bounds[1]), (v1.bounds[2], v1.bounds[1]),
         (v1.bounds[2], v1.bounds[3]), (v1.bounds[0], v1.bounds[3])])) < 1e-6)

    try:
        VoronoiDiagram([(0.0, 0.0), (0.0, 0.0), (5.0, 5.0), (9.0, 1.0)])
        check("rejects duplicate sites", False)
    except ValueError:
        check("rejects duplicate sites", True)

    gj = fast.to_geojson()
    check("geojson rings are closed", all(
        f["geometry"]["coordinates"][0][0] == f["geometry"]["coordinates"][0][-1]
        for f in gj["features"]
    ))


# --------------------------------------------------------------- alpha shape
def test_alpha() -> None:
    section("alpha shape / concave hull")
    rng = random.Random(3)

    # alpha = infinity must reproduce the convex hull exactly.
    pts = [(rng.uniform(0, 500), rng.uniform(0, 500)) for _ in range(120)]
    rings = alpha_shape(pts, alpha_radius=math.inf)
    hull = convex_hull(pts)
    check(
        "alpha=inf reproduces the convex hull",
        abs(polygon_area(rings[0]) - polygon_area(hull)) < 1e-6,
        f"{polygon_area(rings[0]):.2f} vs {polygon_area(hull):.2f}",
    )

    # THE motivating case: two dense clusters separated by a void -- the
    # "river" from the module docstring. The convex hull spans the gap; the
    # alpha shape must not.
    left = [(rng.gauss(0, 30), rng.gauss(0, 30)) for _ in range(150)]
    right = [(rng.gauss(400, 30), rng.gauss(0, 30)) for _ in range(150)]
    both = left + right

    hull_area_ = polygon_area(convex_hull(both))
    rings = alpha_shape(both, alpha_radius=40.0)
    alpha_area = sum(polygon_area(r) for r in rings)

    check(
        f"alpha shape excludes the void (hull {hull_area_:,.0f} m2 -> alpha {alpha_area:,.0f} m2)",
        alpha_area < hull_area_ * 0.5,
    )
    check("disconnected clusters yield separate rings", len(rings) >= 2, f"{len(rings)} rings")

    # A point in the middle of the void: inside the hull, outside the alpha
    # shape. This is precisely the false positive the hull produces.
    mid = (200.0, 0.0)
    in_hull = point_in_polygon(mid, convex_hull(both))
    in_alpha = any(point_in_polygon(mid, r) for r in rings)
    check("void midpoint is inside hull but outside alpha shape", in_hull and not in_alpha)

    # Rings must be well formed.
    check("rings are CCW", all(signed_area(r) > 0 for r in rings))
    check("rings have >= 3 vertices", all(len(r) >= 3 for r in rings))

    # Alpha too small: must degrade to the hull, not return nothing.
    rings = alpha_shape(pts, alpha_radius=1e-6)
    check("over-aggressive alpha falls back to hull", len(rings) == 1 and len(rings[0]) >= 3)

    check("fewer than 3 points", len(alpha_shape([(0.0, 0.0), (1.0, 1.0)])[0]) == 2)

    # The auto-tuned default must produce something sane on clustered data.
    rings = alpha_shape(both)
    check("suggest_alpha default tightens vs hull", sum(polygon_area(r) for r in rings) < hull_area_)


if __name__ == "__main__":
    test_projection()
    test_hull()
    test_pip()
    test_simplify()
    test_delaunay()
    test_voronoi()
    test_alpha()
    print(f"\n{'=' * 46}\n{PASS} passed, {FAIL} failed\n{'=' * 46}")
    sys.exit(1 if FAIL else 0)
