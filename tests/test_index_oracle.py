"""k-d tree range_query vs brute-force oracle."""

from __future__ import annotations

from geogenie.eval.oracle import guarded_voronoi, oracle_index_range, oracle_reachable
import pytest


def test_index_range_oracle():
    oracle_index_range(n_pois=2000, n_queries=50)


def test_reachable_oracle():
    oracle_reachable(n_pois=1500)


def test_voronoi_guard():
    with pytest.raises(ValueError, match="2000"):
        guarded_voronoi([(float(i), float(i)) for i in range(2001)])
