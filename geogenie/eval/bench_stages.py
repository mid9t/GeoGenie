"""Staged latency benchmark: median + IQR, share-based reporting. [VR §4.4]"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import POI
from geogenie.geometry.concave import reachable_area
from geogenie.geometry.hull import convex_hull
from geogenie.geometry.point_in_polygon import point_in_polygon
from geogenie.index.kdtree import KDTreeIndex
from geogenie.reach.pip_fast import pip_filter_vectorized
from geogenie.reach.pipeline import SLACK, reachable_pois
from geogenie.reach.ring_cache import RingCache
from geogenie.routing.accessible_path import WALK_SPEED_M_PER_MIN, walk_frontier
from geogenie.store.ingest import generate_records


def _median_iqr(xs: Sequence[float]) -> Tuple[float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0)
    xs = sorted(xs)
    med = statistics.median(xs)
    q1 = statistics.median(xs[: len(xs) // 2]) if len(xs) > 1 else xs[0]
    q3 = statistics.median(xs[(len(xs) + 1) // 2 :]) if len(xs) > 1 else xs[0]
    return (med, q1, q3)


def _time_fn(fn: Callable[[], None], warmup: int = 5, reps: int = 30) -> List[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return times


def _make_pois(n: int, seed: int = 0) -> Tuple[List[POI], Origin, KDTreeIndex]:
    records = generate_records(n, seed=seed)
    origin = Origin(-122.3, 37.6)
    pois = []
    for r in records:
        x, y = to_xy(r["lon"], r["lat"], origin)
        pois.append(
            POI(
                id=r["id"],
                lon=r["lon"],
                lat=r["lat"],
                x=x,
                y=y,
                category=r.get("category"),
                accessible=bool(r.get("accessible")),
                noise_level=r.get("noise_level"),
            )
        )
    idx = KDTreeIndex(origin=origin)
    idx.build(pois)
    return pois, origin, idx


def bench(n: int = 10_000, reps: int = 30, seed: int = 0) -> Dict:
    pois, ds_origin, idx = _make_pois(n, seed=seed)
    rng = random.Random(seed)
    query_poi = rng.choice(pois)
    origin = Origin(query_poi.lon, query_poi.lat)
    minutes = 10.0
    radius = minutes * WALK_SPEED_M_PER_MIN * SLACK
    qx, qy = to_xy(origin.lon, origin.lat, ds_origin)

    # Stage: index
    def stage_index():
        idx.range_query(qx, qy, radius)

    t_index = _time_fn(stage_index, reps=reps)

    # Stage: ring build cold
    cache = RingCache()

    def stage_ring_cold():
        cache._store.clear()
        frontier = walk_frontier(None, origin, minutes)
        reachable_area(frontier, method="alpha")

    t_ring_cold = _time_fn(stage_ring_cold, reps=max(10, reps // 2))

    # Stage: ring cache hit
    ring = reachable_pois(origin, minutes, idx, ring_cache=cache).ring
    ring_xy = [to_xy(lon, lat, origin) for lon, lat in ring.ring_lonlat]

    def stage_ring_hit():
        cache.get(origin, minutes)

    t_ring_hit = _time_fn(stage_ring_hit, reps=reps)

    candidates = idx.range_query(qx, qy, radius)
    pts = np.array([to_xy(p.lon, p.lat, origin) for p in candidates], dtype=np.float64)
    ring_arr = np.asarray(ring_xy, dtype=np.float64)

    def stage_pip():
        if len(pts):
            pip_filter_vectorized(pts, ring_arr)

    t_pip = _time_fn(stage_pip, reps=reps)

    med = {k: _median_iqr(v) for k, v in {
        "index_ms": t_index,
        "ring_cold_ms": t_ring_cold,
        "ring_hit_ms": t_ring_hit,
        "pip_ms": t_pip,
    }.items()}

    # Shares on cold-path total (index + ring_cold + pip)
    cold_total = med["index_ms"][0] + med["ring_cold_ms"][0] + med["pip_ms"][0]
    shares = {}
    if cold_total > 0:
        shares = {
            "index": med["index_ms"][0] / cold_total,
            "ring_cold": med["ring_cold_ms"][0] / cold_total,
            "pip": med["pip_ms"][0] / cold_total,
        }

    # Sanity gate [VR §4.1]
    examined_frac = len(candidates) / max(len(pois), 1)
    flags = []
    if examined_frac < 0.01 and cold_total > 0:
        # Compare to naive full PIP estimate
        naive_est = med["pip_ms"][0] * (len(pois) / max(len(candidates), 1))
        speedup = naive_est / cold_total if cold_total else 0
        if speedup < 5:
            flags.append(
                f"SANITY: examined {examined_frac:.2%} but speedup~{speedup:.1f}x < 5x; "
                f"shares={shares}"
            )

    return {
        "n": n,
        "candidates": len(candidates),
        "examined_frac": examined_frac,
        "median_iqr_ms": {
            k: {"median": v[0], "q1": v[1], "q3": v[2]} for k, v in med.items()
        },
        "shares_cold": shares,
        "flags": flags,
    }


def overreport(seed: int = 0) -> Dict:
    """Hull vs alpha over-report curve on carved-wedge frontiers. [VR §3.3]"""
    rng = random.Random(seed)
    curves = []
    for angle_deg in range(30, 121, 15):
        # Synthetic wedge frontier
        angle = math.radians(angle_deg)
        pts = [(0.0, 0.0)]
        r = 800.0
        for i in range(40):
            a = -angle / 2 + angle * i / 39
            pts.append((r * math.cos(a), r * math.sin(a)))
        # Fill wedge interior
        for _ in range(80):
            a = rng.uniform(-angle / 2, angle / 2)
            rr = rng.uniform(0, r)
            pts.append((rr * math.cos(a), rr * math.sin(a)))
        hull = convex_hull(pts)
        alpha = reachable_area(pts, method="alpha")
        # Random POIs in bbox
        pois = [(rng.uniform(-r, r), rng.uniform(-r, r)) for _ in range(2000)]
        hull_ids = {i for i, p in enumerate(pois) if point_in_polygon(p, hull)}
        alpha_ids = {i for i, p in enumerate(pois) if point_in_polygon(p, alpha)}
        assert hull_ids >= alpha_ids, "hull must be admissible over-approx"
        over = (
            (len(hull_ids) - len(alpha_ids)) / len(alpha_ids) if alpha_ids else float("inf")
        )
        curves.append(
            {
                "wedge_angle_deg": angle_deg,
                "hull_count": len(hull_ids),
                "alpha_count": len(alpha_ids),
                "over_report": over,
                "label": "synthetic",
            }
        )
    return {"curve": curves}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--overreport", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args(argv)
    if args.overreport:
        report = overreport()
    else:
        report = bench(n=args.n, reps=args.reps)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
