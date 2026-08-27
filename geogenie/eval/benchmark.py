"""
Multi-size, multi-index latency compare harness.

This file owns INDEX STAGE timing (k-d / brute / R-tree). Stage decomposition
inside the reach pipeline lives in bench_stages.py — do not consolidate them;
they answer different questions. [VR §4 / BENCHMARK_GAP Step 3]

CLI (README contract):
  python -m geogenie.eval.benchmark --sizes 1000 10000 100000 \\
      --compare kdtree brute_force rtree --out eval/results/index_compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Type

from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import POI
from geogenie.index.brute import BruteForceIndex
from geogenie.index.kdtree import KDTreeIndex, brute_force_range
from geogenie.index.rtree import RTreeIndex
from geogenie.store.ingest import generate_records

INDEXES: Dict[str, Type] = {
    "kdtree": KDTreeIndex,
    "brute_force": BruteForceIndex,
    "rtree": RTreeIndex,
}


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


def _median_iqr(xs: Sequence[float]) -> Dict[str, Any]:
    xs = sorted(xs)
    if not xs:
        return {"median": 0.0, "iqr": [0.0, 0.0]}
    med = statistics.median(xs)
    q1 = statistics.median(xs[: len(xs) // 2]) if len(xs) > 1 else xs[0]
    q3 = statistics.median(xs[(len(xs) + 1) // 2 :]) if len(xs) > 1 else xs[0]
    return {"median": med, "iqr": [q1, q3]}


def make_pois(n: int, seed: int) -> Tuple[List[POI], Origin]:
    records = generate_records(n, seed=seed)
    origin = Origin(-122.3, 37.6)
    pois = []
    for r in records:
        x, y = to_xy(r["lon"], r["lat"], origin)
        pois.append(POI(id=r["id"], lon=r["lon"], lat=r["lat"], x=x, y=y))
    return pois, origin


def sample_origins(
    pois: Sequence[POI], n_queries: int, seed: int
) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    return [(p.x or 0.0, p.y or 0.0) for p in rng.sample(list(pois), min(n_queries, len(pois)))]


def correctness_gate(
    indices: Dict[str, Any], pois: List[POI], origins: List[Tuple[float, float]], radius_m: float
) -> None:
    """Abort before timing if any index disagrees with the oracle."""
    for ox, oy in origins:
        exp = {p.id for p in brute_force_range(pois, ox, oy, radius_m)}
        for name, idx in indices.items():
            got = set(idx.range_query(ox, oy, radius_m))
            if got != exp:
                diff = got.symmetric_difference(exp)
                raise AssertionError(
                    f"CORRECTNESS GATE FAILED for {name}: "
                    f"|symdiff|={len(diff)} sample={list(diff)[:10]}"
                )


def run(
    sizes: Sequence[int],
    names: Sequence[str],
    queries: int,
    reps: int,
    warmup: int,
    radius_m: float,
    seed: int,
) -> Dict[str, Any]:
    results = []
    warnings = []
    for n in sizes:
        pois, origin = make_pois(n, seed)
        origins = sample_origins(pois, queries, seed + n)
        built = {}
        for name in names:
            cls = INDEXES[name]
            idx = cls(origin=origin)
            t0 = time.perf_counter_ns()
            idx.build(pois)
            build_ns = time.perf_counter_ns() - t0
            built[name] = (idx, build_ns)

        correctness_gate({k: v[0] for k, v in built.items()}, pois, origins, radius_m)

        # Reference examined fraction from kdtree (or first index)
        ref_name = "kdtree" if "kdtree" in built else names[0]
        ref_idx = built[ref_name][0]
        examined = []
        for ox, oy in origins:
            examined.append(len(ref_idx.range_query(ox, oy, radius_m)) / max(n, 1))
        examined_frac = statistics.median(examined) if examined else 0.0

        brute_query_us = None
        for name in names:
            idx, build_ns = built[name]
            # Separate build timing samples (rebuild each sample for IQR)
            build_samples_ms = []
            for _ in range(warmup + max(5, min(reps, 15))):
                fresh = INDEXES[name](origin=origin)
                t0 = time.perf_counter_ns()
                fresh.build(pois)
                build_samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
            build_samples_ms = build_samples_ms[warmup:]

            query_batch_us = []  # µs per query, one value per rep
            for r in range(warmup + reps):
                t0 = time.perf_counter_ns()
                for ox, oy in origins:
                    idx.range_query(ox, oy, radius_m)
                elapsed_ns = time.perf_counter_ns() - t0
                if r >= warmup:
                    query_batch_us.append((elapsed_ns / 1e3) / max(len(origins), 1))

            qstats = _median_iqr(query_batch_us)
            row = {
                "n": n,
                "index": name,
                "build_ms": _median_iqr(build_samples_ms),
                "query_us_per_query": qstats,
                "candidates_examined_frac": examined_frac if name == ref_name else None,
            }
            results.append(row)
            if name == "brute_force":
                brute_query_us = qstats["median"]

        # Self-consistency gate at large n
        if n >= 100_000 and brute_query_us and examined_frac < 0.01:
            for row in results:
                if row["n"] != n or row["index"] == "brute_force":
                    continue
                speedup = brute_query_us / max(row["query_us_per_query"]["median"], 1e-12)
                if speedup < 5:
                    warnings.append(
                        f"SANITY at n={n} {row['index']}: examined {examined_frac:.2%} "
                        f"but speedup {speedup:.1f}x < 5× vs brute — check harness "
                        f"(hydration leak?) or see bench_stages.py for Amdahl story"
                    )

    return {
        "env": _env(),
        "config": {
            "sizes": list(sizes),
            "compare": list(names),
            "queries": queries,
            "reps": reps,
            "warmup": warmup,
            "radius_m": radius_m,
            "seed": seed,
        },
        "results": results,
        "warnings": warnings,
    }


def markdown_table(report: Dict[str, Any]) -> str:
    by_n: Dict[int, Dict[str, float]] = {}
    for row in report["results"]:
        by_n.setdefault(row["n"], {})[row["index"]] = row["query_us_per_query"]["median"]
    lines = [
        "| Dataset size | Brute-force (µs/q) | k-d tree (µs/q) | R-tree (µs/q) | Speedup (best vs brute) |",
        "|---|---|---|---|---|",
    ]
    for n in sorted(by_n):
        cells = by_n[n]
        brute = cells.get("brute_force")
        kd = cells.get("kdtree")
        rt = cells.get("rtree")

        def fmt(v):
            return f"{v:.2f}" if v is not None else "—"

        best = None
        for name in ("kdtree", "rtree"):
            if name in cells and brute:
                sp = brute / max(cells[name], 1e-12)
                best = sp if best is None else max(best, sp)
        lines.append(
            f"| {n:,} | {fmt(brute)} | {fmt(kd)} | {fmt(rt)} | "
            f"{f'{best:.1f}×' if best else '—'} |"
        )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Index compare harness")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    p.add_argument(
        "--compare",
        nargs="+",
        default=["kdtree", "brute_force", "rtree"],
        choices=list(INDEXES),
    )
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--radius-m", type=float, default=800.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="eval/results/index_compare.json")
    args = p.parse_args(argv)

    report = run(
        sizes=args.sizes,
        names=args.compare,
        queries=args.queries,
        reps=args.reps,
        warmup=args.warmup,
        radius_m=args.radius_m,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(markdown_table(report))
    for w in report.get("warnings", []):
        print("WARNING:", w)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
