"""Oracle-based correctness checks (run in CI)."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import List, Sequence, Set, Tuple

import numpy as np

from geogenie.core.coords import Origin, haversine, to_xy
from geogenie.core.types import POI
from geogenie.geometry.point_in_polygon import point_in_polygon
from geogenie.index.kdtree import KDTreeIndex, brute_force_range
from geogenie.reach.pip_fast import pip_filter_vectorized
from geogenie.reach.pipeline import reachable_pois
from geogenie.reach.ring_cache import RingCache
from geogenie.store.ingest import generate_records


def _pois_from_records(records, origin: Origin) -> List[POI]:
    pois = []
    for r in records:
        x, y = to_xy(r["lon"], r["lat"], origin)
        pois.append(
            POI(
                id=r["id"],
                lon=r["lon"],
                lat=r["lat"],
                category=r.get("category"),
                name=r.get("name"),
                accessible=bool(r.get("accessible")),
                noise_level=r.get("noise_level"),
                hours=r.get("hours"),
                x=x,
                y=y,
            )
        )
    return pois


def oracle_index_range(n_pois: int = 5000, n_queries: int = 50, seed: int = 0) -> None:
    rng = random.Random(seed)
    records = generate_records(n_pois, seed=seed)
    origin = Origin(-122.3, 37.6)
    pois = _pois_from_records(records, origin)
    idx = KDTreeIndex(origin=origin)
    idx.build(pois)
    for _ in range(n_queries):
        p = rng.choice(pois)
        radius = rng.uniform(200, 2000)
        got = {p.id for p in idx.range_query(p.x or 0, p.y or 0, radius)}
        exp = {p.id for p in brute_force_range(pois, p.x or 0, p.y or 0, radius)}
        assert got == exp, (len(got), len(exp), got.symmetric_difference(exp))


def oracle_pip_fast(n_points: int = 2000, seed: int = 1) -> None:
    rng = random.Random(seed)
    # Unit square ring + U-notch style polygon
    ring = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (6.0, 10.0),
        (6.0, 4.0),
        (4.0, 4.0),
        (4.0, 10.0),
        (0.0, 10.0),
    ]
    pts = [(rng.uniform(-1, 11), rng.uniform(-1, 11)) for _ in range(n_points)]
    # Degeneracies: vertices, edge midpoints
    pts.extend(ring)
    pts.extend([((ring[i][0] + ring[(i + 1) % len(ring)][0]) / 2,
                 (ring[i][1] + ring[(i + 1) % len(ring)][1]) / 2) for i in range(len(ring))])
    mask = pip_filter_vectorized(
        np.asarray(pts, dtype=np.float64),
        np.asarray(ring, dtype=np.float64),
    )
    for i, p in enumerate(pts):
        ref = point_in_polygon(p, ring, include_boundary=True)
        assert bool(mask[i]) == ref, (i, p, bool(mask[i]), ref)


def oracle_to_xy(seed: int = 2) -> None:
    rng = random.Random(seed)
    for lat0 in (0.0, 40.0, 60.0, 69.0):
        o = Origin(0.0, lat0)
        for _ in range(500):
            dx = rng.uniform(-10_000, 10_000)
            dy = rng.uniform(-10_000, 10_000)
            from geogenie.core.coords import to_lonlat

            lon, lat = to_lonlat(dx, dy, o)
            plane = math.hypot(dx, dy)
            if plane < 1:
                continue
            hav = haversine(lat0, 0.0, lat, lon)
            assert abs(plane - hav) / hav < 0.006
    for bad in (71.0, 80.0, 91.0):
        try:
            to_xy(0.0, bad, Origin(0.0, 40.0))
            raise AssertionError(f"expected raise at lat={bad}")
        except ValueError:
            pass


def oracle_reachable(n_pois: int = 2000, seed: int = 3) -> None:
    records = generate_records(n_pois, seed=seed)
    origin_ds = Origin(-122.3, 37.6)
    pois = _pois_from_records(records, origin_ds)
    idx = KDTreeIndex(origin=origin_ds)
    idx.build(pois)
    rng = random.Random(seed)
    for _ in range(20):
        base = rng.choice(pois)
        o = Origin(base.lon, base.lat)
        for minutes in (5.0, 10.0, 15.0):
            # Fresh cache each time for oracle (no cache pollution)
            got = reachable_pois(o, minutes, idx, ring_cache=RingCache())
            # Brute: all POIs + scalar PIP against same ring
            ring_xy = [to_xy(lon, lat, o) for lon, lat in got.ring.ring_lonlat]
            exp_ids = set()
            for p in pois:
                xy = to_xy(p.lon, p.lat, o)
                if point_in_polygon(xy, ring_xy, include_boundary=True):
                    exp_ids.add(p.id)
            got_ids = {p.id for p in got.pois}
            assert got_ids == exp_ids, (
                minutes,
                len(got_ids),
                len(exp_ids),
                len(got_ids.symmetric_difference(exp_ids)),
            )


def guarded_voronoi(sites: Sequence[Tuple[float, float]], margin: float = 500.0):
    """Voronoi only on viewport survivors; raise if n > 2000. [VR §4.5]"""
    if len(sites) > 2000:
        raise ValueError(
            f"Voronoi refused for n={len(sites)} > 2000; prefilter with k-d tree first"
        )
    from geogenie.geometry.voronoi import VoronoiDiagram

    return VoronoiDiagram(list(sites), margin=margin)


def run_all() -> None:
    print("oracle_index_range...")
    oracle_index_range()
    print("oracle_pip_fast...")
    oracle_pip_fast()
    print("oracle_to_xy...")
    oracle_to_xy()
    print("oracle_reachable...")
    oracle_reachable()
    print("all oracles passed")


if __name__ == "__main__":
    run_all()
