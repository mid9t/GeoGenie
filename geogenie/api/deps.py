"""FastAPI lifespan state: DB → index → ring cache, built once at startup."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from geogenie.core.coords import Origin
from geogenie.index.kdtree import KDTreeIndex
from geogenie.reach.ring_cache import RingCache
from geogenie.store.db import load_pois
from geogenie.store.ingest import generate_records, write_pois


@dataclass
class AppState:
    db_path: str
    index: KDTreeIndex
    ring_cache: RingCache = field(default_factory=RingCache)
    n_pois: int = 0


_STATE: Optional[AppState] = None


def get_state() -> AppState:
    if _STATE is None:
        raise RuntimeError("app state not initialized")
    return _STATE


def init_state(
    db_path: Optional[str] = None,
    *,
    bootstrap_n: int = 5000,
) -> AppState:
    global _STATE
    path = db_path or os.environ.get("GEOGENIE_DB", "data/pois.db")
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        write_pois(p, generate_records(bootstrap_n, seed=42))
    pois = load_pois(p)
    # Index origin = centroid of dataset
    if pois:
        origin = Origin(
            sum(x.lon for x in pois) / len(pois),
            sum(x.lat for x in pois) / len(pois),
        )
    else:
        origin = Origin(-122.42, 37.77)
    index = KDTreeIndex(origin=origin)
    index.build(pois)
    _STATE = AppState(db_path=str(p), index=index, n_pois=len(pois))
    return _STATE
