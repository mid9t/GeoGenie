"""Ingest CHECK constraints and lon/lat validation."""

from __future__ import annotations

import pytest

from geogenie.store.db import load_pois
from geogenie.store.ingest import generate_records, validate_lonlat, write_pois


def test_validate_rejects_bad_lat():
    with pytest.raises(ValueError):
        validate_lonlat(-122.4, -122.4)  # swapped SF


def test_ingest_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    recs = generate_records(100, seed=1)
    n = write_pois(db, recs)
    assert n == 100
    pois = load_pois(db)
    assert len(pois) == 100
    assert all(-180 <= p.lon <= 180 and -90 <= p.lat <= 90 for p in pois)


def test_ingest_rejects_swap(tmp_path):
    db = tmp_path / "bad.db"
    with pytest.raises(ValueError):
        write_pois(
            db,
            [{"id": 1, "lon": 37.77, "lat": -122.42, "name": "x", "category": "cafe",
              "accessible": 1, "noise_level": "quiet", "hours": "{}"}],
        )
