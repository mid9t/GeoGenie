"""BruteForceIndex oracle set-equality vs KDTreeIndex."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import POI
from geogenie.index.brute import BruteForceIndex
from geogenie.index.kdtree import KDTreeIndex, brute_force_range
from geogenie.store.ingest import generate_records


def _pois(n: int, seed: int, origin: Origin) -> list[POI]:
    records = generate_records(n, seed=seed)
    out = []
    for r in records:
        x, y = to_xy(r["lon"], r["lat"], origin)
        out.append(
            POI(
                id=r["id"],
                lon=r["lon"],
                lat=r["lat"],
                x=x,
                y=y,
                category=r.get("category"),
            )
        )
    return out


@pytest.mark.parametrize("n", [500, 5000])
def test_brute_matches_oracle(n):
    origin = Origin(-122.3, 37.6)
    pois = _pois(n, seed=7, origin=origin)
    idx = BruteForceIndex(origin=origin)
    idx.build(pois)
    rng = random.Random(7)
    for _ in range(50):
        p = rng.choice(pois)
        radius = rng.uniform(200, 2000)
        got = set(idx.range_query(p.x or 0, p.y or 0, radius))
        exp = {q.id for q in brute_force_range(pois, p.x or 0, p.y or 0, radius)}
        assert got == exp


def test_brute_matches_kdtree_large():
    origin = Origin(-122.3, 37.6)
    # 20k is enough for CI; 120k covered in benchmark correctness gate
    pois = _pois(20_000, seed=11, origin=origin)
    kd = KDTreeIndex(origin=origin)
    bf = BruteForceIndex(origin=origin)
    kd.build(pois)
    bf.build(pois)
    rng = random.Random(11)
    for _ in range(50):
        p = rng.choice(pois)
        radius = rng.uniform(300, 1500)
        a = set(kd.range_query(p.x or 0, p.y or 0, radius))
        b = set(bf.range_query(p.x or 0, p.y or 0, radius))
        assert a == b


def test_k_nearest_agrees():
    origin = Origin(-122.3, 37.6)
    pois = _pois(1000, seed=3, origin=origin)
    kd = KDTreeIndex(origin=origin)
    bf = BruteForceIndex(origin=origin)
    kd.build(pois)
    bf.build(pois)
    p = pois[0]
    assert kd.k_nearest(p.x or 0, p.y or 0, 5) == bf.k_nearest(p.x or 0, p.y or 0, 5)
