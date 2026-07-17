"""Ingest synthetic or CSV POIs into the SQLite schema with CHECK constraints."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from geogenie.store.db import connect, init_schema

CITY_CENTERS = [
    ("San Francisco", -122.4194, 37.7749, 20),
    ("Daly City", -122.4702, 37.6879, 4),
    ("Oakland", -122.2711, 37.8044, 10),
    ("Berkeley", -122.2727, 37.8715, 5),
    ("San Mateo", -122.3255, 37.5630, 5),
    ("Palo Alto", -122.1430, 37.4419, 6),
    ("Mountain View", -122.0839, 37.3861, 6),
    ("Sunnyvale", -122.0363, 37.3688, 7),
    ("San Jose", -121.8863, 37.3382, 18),
]

CATEGORIES = [
    ("restaurant", 16),
    ("cafe", 12),
    ("retail", 14),
    ("grocery", 7),
    ("bar", 6),
    ("park", 5),
    ("pharmacy", 4),
    ("library", 2),
    ("gym", 5),
    ("museum", 2),
]

NOISE_LEVELS = ["quiet", "moderate", "loud", "very_loud"]
LOUD = {"bar", "gym"}
QUIET = {"library", "park", "museum"}


def _weighted_choice(rng: random.Random, items: Sequence[tuple]) -> Any:
    total = sum(w for _, w in items)
    r = rng.uniform(0, total)
    upto = 0.0
    for value, weight in items:
        upto += weight
        if upto >= r:
            return value
    return items[-1][0]


def validate_lonlat(lon: float, lat: float) -> None:
    """Reject swapped / out-of-range coords at ingest (DB CHECK twin)."""
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"invalid lat {lat} (possible lon/lat swap)")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"invalid lon {lon}")
    # Heuristic: Bay Area lon is negative; lat ~37. If someone swapped SF coords
    # they'd get lat≈-122 which already fails the lat CHECK. Extra belt:
    if abs(lon) < 1.0 and abs(lat) > 50:
        raise ValueError(f"suspicious (lon, lat)=({lon}, {lat}); possible swap")


def generate_records(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    weighted_cities = [((name, lon, lat), w) for name, lon, lat, w in CITY_CENTERS]
    for i in range(1, n + 1):
        city_name, lon0, lat0 = _weighted_choice(rng, weighted_cities)
        lon = lon0 + rng.gauss(0, 0.022)
        lat = lat0 + rng.gauss(0, 0.020)
        lon = max(-180.0, min(180.0, lon))
        lat = max(-90.0, min(90.0, lat))
        validate_lonlat(lon, lat)
        category = _weighted_choice(rng, CATEGORIES)
        if category in LOUD:
            noise = _weighted_choice(rng, list(zip(NOISE_LEVELS, [1, 3, 8, 6])))
        elif category in QUIET:
            noise = _weighted_choice(rng, list(zip(NOISE_LEVELS, [8, 6, 1, 0.2])))
        else:
            noise = _weighted_choice(rng, list(zip(NOISE_LEVELS, [3, 6, 3, 1])))
        records.append(
            {
                "id": i,
                "lon": round(lon, 6),
                "lat": round(lat, 6),
                "name": f"{city_name} {category.title()} {i}",
                "category": category,
                "accessible": int(rng.random() < 0.55),
                "noise_level": noise,
                "hours": json.dumps({"mon": {"open": "08:00", "close": "18:00"}}),
            }
        )
    return records


def write_pois(db_path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = connect(path)
    try:
        init_schema(conn)
        count = 0
        for r in records:
            validate_lonlat(float(r["lon"]), float(r["lat"]))
            try:
                conn.execute(
                    """
                    INSERT INTO pois (id, lon, lat, name, category, accessible, noise_level, hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(r["id"]),
                        float(r["lon"]),
                        float(r["lat"]),
                        r.get("name"),
                        r.get("category"),
                        int(r["accessible"]) if r.get("accessible") is not None else None,
                        r.get("noise_level"),
                        r.get("hours") if isinstance(r.get("hours"), str) else json.dumps(r.get("hours")),
                    ),
                )
                count += 1
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"ingest rejected row {r.get('id')}: {exc}") from exc
        conn.commit()
        return count
    finally:
        conn.close()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate and ingest synthetic POIs")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/pois.db")
    args = parser.parse_args(argv)
    records = generate_records(args.n, args.seed)
    n = write_pois(args.out, records)
    print(f"Wrote {n:,} POIs to {args.out}")


if __name__ == "__main__":
    main()
