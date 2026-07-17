"""Stage A: coordinate funnel guards and haversine agreement."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

import pytest

from geogenie.core.coords import MAX_SAFE_LAT, Origin, haversine, to_lonlat, to_xy

ROOT = Path(__file__).resolve().parents[1]


def test_origin_identity():
    o = Origin(-122.4194, 37.7749)
    assert to_xy(o.lon, o.lat, o) == (0.0, 0.0)


def test_roundtrip():
    o = Origin(-122.42, 37.77)
    x, y = to_xy(-122.41, 37.78, o)
    lon, lat = to_lonlat(x, y, o)
    assert abs(lon - (-122.41)) < 1e-9
    assert abs(lat - 37.78) < 1e-9


@pytest.mark.parametrize("lat", [71.0, 80.0, 91.0, -71.0, -91.0])
def test_raises_beyond_guard(lat):
    o = Origin(0.0, 40.0)
    with pytest.raises(ValueError):
        to_xy(0.0, lat, o)


def test_raises_invalid_origin_lat():
    with pytest.raises(ValueError):
        to_xy(0.0, 40.0, Origin(0.0, 75.0))


def test_haversine_agreement_within_guard():
    rng = random.Random(0)
    for lat0 in (0.0, 40.0, 60.0, 69.0):
        o = Origin(0.0, lat0)
        for _ in range(500):
            # random point within ~15 km
            dx = rng.uniform(-10_000, 10_000)
            dy = rng.uniform(-10_000, 10_000)
            lon, lat = to_lonlat(dx, dy, o)
            dist_plane = math.hypot(dx, dy)
            if dist_plane < 1.0:
                continue
            dist_hav = haversine(lat0, 0.0, lat, lon)
            rel = abs(dist_plane - dist_hav) / dist_hav
            assert rel < 0.006, (lat0, dist_plane, dist_hav, rel)


def test_only_coords_imports_projection():
    """CI grep: geometry.projection only imported via core/coords."""
    pattern = re.compile(
        r"from\s+geogenie\.geometry\.projection\s+import|import\s+geogenie\.geometry\.projection"
    )
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = str(path.relative_to(ROOT))
        if rel.startswith("geogenie/core/coords.py"):
            continue
        if rel.startswith("geogenie/geometry/"):
            continue
        if "__pycache__" in rel or ".venv" in rel:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            offenders.append(rel)
    assert offenders == [], f"projection imported outside core/coords: {offenders}"


def test_max_safe_lat_constant():
    assert MAX_SAFE_LAT == 70.0
