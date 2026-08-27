"""
End-to-end reachability latency: cold vs warm p50/p95.

Mode A (default): in-process reachable_pois — deterministic geometry path.
Mode B (--http): POST /search/structured via httpx — includes FastAPI/serde.

LLM parse time is excluded from both modes (network-bound; see parser_eval).
Do not mix Mode A and Mode B numbers in one README table row.

Stage decomposition of one pipeline query lives in bench_stages.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import POI
from geogenie.index.kdtree import KDTreeIndex
from geogenie.reach.pipeline import reachable_pois
from geogenie.reach.ring_cache import RingCache
from geogenie.store.ingest import generate_records


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _env() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
    }


def _make_pois(n: int, seed: int) -> Tuple[List[POI], Origin, KDTreeIndex]:
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


def _zipf_origins(
    pois: Sequence[POI], n_origins: int, n_requests: int, zipf: float, seed: int
) -> List[Origin]:
    """Sample request origins from a Zipf-weighted pool (exponent zipf)."""
    rng = random.Random(seed)
    pool = [Origin(p.lon, p.lat) for p in rng.sample(list(pois), min(n_origins, len(pois)))]
    # ranks 1..m ; P(i) ∝ i^(-zipf)
    m = len(pool)
    weights = np.array([(i + 1) ** (-zipf) for i in range(m)], dtype=np.float64)
    weights /= weights.sum()
    choices = rng.choices(pool, weights=weights.tolist(), k=n_requests)
    return choices


def _percentiles(samples_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    p50, p95 = np.percentile(arr, [50, 95])
    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "n": int(len(arr)),
        "min_ms": float(arr.min()) if len(arr) else 0.0,
        "max_ms": float(arr.max()) if len(arr) else 0.0,
    }


def bench_inprocess(
    n_pois: int,
    requests: int,
    n_origins: int,
    zipf: float,
    minutes: float,
    seed: int,
) -> Dict[str, Any]:
    pois, _, idx = _make_pois(n_pois, seed)
    origins = _zipf_origins(pois, n_origins, requests, zipf, seed)

    # Cold: no reused cache — each request builds its own ring
    cold_ms = []
    for o in origins:
        t0 = time.perf_counter()
        reachable_pois(o, minutes, idx, ring_cache=RingCache())
        cold_ms.append((time.perf_counter() - t0) * 1000.0)

    # Warm: one shared cache, prime then measure
    warm_cache = RingCache(maxsize=256)
    for o in set(origins):  # prime unique origins
        reachable_pois(o, minutes, idx, ring_cache=warm_cache)
    warm_ms = []
    for o in origins:
        t0 = time.perf_counter()
        reachable_pois(o, minutes, idx, ring_cache=warm_cache)
        warm_ms.append((time.perf_counter() - t0) * 1000.0)

    cold = _percentiles(cold_ms)
    warm = _percentiles(warm_ms)
    if not (warm["p50_ms"] < cold["p50_ms"]):
        raise RuntimeError(
            f"warm p50 ({warm['p50_ms']:.2f}) not < cold p50 ({cold['p50_ms']:.2f}); "
            "cache or Zipf workload broken — investigate before publishing"
        )
    return {
        "mode": "in-process",
        "cold": cold,
        "warm": warm,
        "workload": {
            "requests": requests,
            "origins_pool": n_origins,
            "zipf": zipf,
            "minutes": minutes,
            "n_pois": n_pois,
            "distribution": "zipf over origin pool",
            "llm_parse_excluded": True,
        },
        "env": _env(),
    }


def bench_http(
    base_url: str,
    requests: int,
    n_origins: int,
    zipf: float,
    minutes: float,
    seed: int,
    n_pois: int,
) -> Dict[str, Any]:
    import httpx

    # Origins from the same synthetic distribution (lat/lon only)
    pois, _, _ = _make_pois(min(n_pois, 5000), seed)
    origins = _zipf_origins(pois, n_origins, requests, zipf, seed)
    radius_m = minutes * 80.0

    def one(client: httpx.Client, o: Origin) -> float:
        t0 = time.perf_counter()
        r = client.post(
            f"{base_url.rstrip('/')}/search/structured",
            json={
                "origin": {"lon": o.lon, "lat": o.lat},
                "radius_m": radius_m,
                "minutes": minutes,
                "filters": {},
            },
            timeout=60.0,
        )
        r.raise_for_status()
        return (time.perf_counter() - t0) * 1000.0

    with httpx.Client() as client:
        # Cold-ish: no client-side cache; server may warm its ring cache — report as http
        samples = [one(client, o) for o in origins]
        # Prime server cache
        for o in set(origins):
            one(client, o)
        warm = [one(client, o) for o in origins]

    return {
        "mode": "http",
        "cold": _percentiles(samples),
        "warm": _percentiles(warm),
        "workload": {
            "requests": requests,
            "origins_pool": n_origins,
            "zipf": zipf,
            "minutes": minutes,
            "base_url": base_url,
            "distribution": "zipf over origin pool",
            "llm_parse_excluded": True,
            "note": "HTTP cold/warm approximate server ring-cache state",
        },
        "env": _env(),
    }


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--origins", type=int, default=20)
    p.add_argument("--zipf", type=float, default=1.2)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--n-pois", type=int, default=10_000)
    p.add_argument("--http", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="eval/results/e2e.json")
    args = p.parse_args(argv)

    if args.http:
        report = bench_http(
            args.http, args.requests, args.origins, args.zipf, args.minutes, args.seed, args.n_pois
        )
    else:
        report = bench_inprocess(
            args.n_pois, args.requests, args.origins, args.zipf, args.minutes, args.seed
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("mode", "cold", "warm")}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
