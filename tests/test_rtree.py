"""R-tree correctness: oracles, degeneracies, property tests."""

from __future__ import annotations

import random
import time

import pytest

from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import POI
from geogenie.index.kdtree import brute_force_range
from geogenie.index.rtree import RTreeIndex
from geogenie.store.ingest import generate_records


def _pois(n: int, seed: int, origin: Origin) -> list[POI]:
    records = generate_records(n, seed=seed)
    out = []
    for r in records:
        x, y = to_xy(r["lon"], r["lat"], origin)
        out.append(POI(id=r["id"], lon=r["lon"], lat=r["lat"], x=x, y=y))
    return out


def _assert_range_eq(idx: RTreeIndex, pois: list[POI], queries: int, seed: int):
    rng = random.Random(seed)
    for _ in range(queries):
        p = rng.choice(pois)
        radius = rng.uniform(0, 2000)
        got = set(idx.range_query(p.x or 0, p.y or 0, radius))
        exp = {q.id for q in brute_force_range(pois, p.x or 0, p.y or 0, radius)}
        assert got == exp, got.symmetric_difference(exp)


@pytest.mark.parametrize("n", [100, 5000])
def test_rtree_property(n):
    origin = Origin(-122.3, 37.6)
    pois = _pois(n, seed=21, origin=origin)
    idx = RTreeIndex(origin=origin)
    idx.build(pois)
    _assert_range_eq(idx, pois, queries=200, seed=21)


def test_rtree_duplicates_and_collinear():
    origin = Origin(-122.3, 37.6)
    pois = []
    # duplicates
    for i in range(10):
        x, y = to_xy(-122.3, 37.6, origin)
        pois.append(POI(id=i, lon=-122.3, lat=37.6, x=x, y=y))
    # collinear along x
    for i in range(10, 40):
        lon = -122.3 + (i - 10) * 0.001
        x, y = to_xy(lon, 37.6, origin)
        pois.append(POI(id=i, lon=lon, lat=37.6, x=x, y=y))
    idx = RTreeIndex(origin=origin, leaf_capacity=8)
    idx.build(pois)
    _assert_range_eq(idx, pois, queries=50, seed=5)


def test_rtree_small_n_root_is_leaf():
    origin = Origin(-122.3, 37.6)
    pois = _pois(10, seed=1, origin=origin)
    idx = RTreeIndex(origin=origin, leaf_capacity=32)
    idx.build(pois)
    assert idx.root is not None and idx.root.is_leaf
    _assert_range_eq(idx, pois, queries=20, seed=1)


def test_rtree_outside_and_radius_zero():
    origin = Origin(-122.3, 37.6)
    pois = _pois(200, seed=2, origin=origin)
    idx = RTreeIndex(origin=origin)
    idx.build(pois)
    # Far away — empty
    assert idx.range_query(1e9, 1e9, 10.0) == []
    # Radius 0 at a known point — at least that id (and any duplicates)
    p = pois[0]
    got = set(idx.range_query(p.x or 0, p.y or 0, 0.0))
    assert p.id in got


def test_rtree_build_time_reasonable():
    origin = Origin(-122.3, 37.6)
    pois = _pois(50_000, seed=9, origin=origin)
    idx = RTreeIndex(origin=origin)
    t0 = time.perf_counter()
    idx.build(pois)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"STR build too slow: {elapsed:.2f}s"
