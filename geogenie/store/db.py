"""SQLite access layer. DB is source of truth; the k-d tree is derived."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geogenie.core.types import POI

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pois (
  id INTEGER PRIMARY KEY,
  lon REAL NOT NULL,
  lat REAL NOT NULL,
  name TEXT,
  category TEXT,
  accessible INTEGER,
  noise_level TEXT,
  hours TEXT,
  CHECK (lat BETWEEN -90 AND 90),
  CHECK (lon BETWEEN -180 AND 180)
);
CREATE TABLE IF NOT EXISTS graph_edges (
  node_a INTEGER,
  node_b INTEGER,
  length_m REAL,
  step_free INTEGER
);
CREATE INDEX IF NOT EXISTS idx_pois_lonlat ON pois(lon, lat);
CREATE INDEX IF NOT EXISTS idx_pois_category ON pois(category);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def _row_to_poi(row: sqlite3.Row) -> POI:
    hours = row["hours"]
    return POI(
        id=int(row["id"]),
        lon=float(row["lon"]),
        lat=float(row["lat"]),
        name=row["name"],
        category=row["category"],
        accessible=bool(row["accessible"]) if row["accessible"] is not None else None,
        noise_level=row["noise_level"],
        hours=hours,
    )


def load_pois(db_path: str | Path) -> List[POI]:
    """Startup: full load into the in-memory spatial index."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, lon, lat, name, category, accessible, noise_level, hours FROM pois"
        ).fetchall()
        return [_row_to_poi(r) for r in rows]
    finally:
        conn.close()


def get_pois_by_ids(db_path: str | Path, ids: Sequence[int]) -> List[POI]:
    """Hydrate attributes at rank time."""
    if not ids:
        return []
    conn = connect(db_path)
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, lon, lat, name, category, accessible, noise_level, hours "
            f"FROM pois WHERE id IN ({placeholders})",
            list(ids),
        ).fetchall()
        by_id = {int(r["id"]): _row_to_poi(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]
    finally:
        conn.close()


def load_street_graph(db_path: str | Path) -> Dict[str, Any]:
    """Phase 4: load edge list. Returns adjacency dict."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT node_a, node_b, length_m, step_free FROM graph_edges"
        ).fetchall()
    except sqlite3.OperationalError:
        return {"nodes": {}, "edges": []}
    finally:
        conn.close()

    adj: Dict[int, List[Tuple[int, float, bool]]] = {}
    edges = []
    for r in rows:
        a, b = int(r["node_a"]), int(r["node_b"])
        length = float(r["length_m"])
        step_free = bool(r["step_free"])
        edges.append((a, b, length, step_free))
        adj.setdefault(a, []).append((b, length, step_free))
        adj.setdefault(b, []).append((a, length, step_free))
    return {"nodes": adj, "edges": edges}
