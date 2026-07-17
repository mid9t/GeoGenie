"""Vectorized PIP vs scalar geometry oracle."""

from __future__ import annotations

import numpy as np

from geogenie.eval.oracle import oracle_pip_fast
from geogenie.geometry.point_in_polygon import point_in_polygon
from geogenie.reach.pip_fast import pip_filter_vectorized


def test_oracle_pip_fast():
    oracle_pip_fast(n_points=3000)


def test_square_and_outside():
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = np.array([[0.5, 0.5], [2.0, 2.0], [0.0, 0.0], [1.0, 0.5]])
    mask = pip_filter_vectorized(pts, ring)
    for i, p in enumerate(pts):
        assert bool(mask[i]) == point_in_polygon(tuple(p), [tuple(r) for r in ring])
