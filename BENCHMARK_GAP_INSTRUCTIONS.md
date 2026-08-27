# GeoGenie — Closing the README ↔ Code Gap

Coding instructions for the five diagnosed gaps. Goal: every command and every table in the README becomes runnable/fillable, with no placeholder cells. Ordered so each step unblocks the next.

**Deliverables at a glance**

| # | Gap | New/changed artifact |
|---|---|---|
| 1 | Brute force is oracle-only, never timed | `index/brute.py` — `BruteForceIndex` behind `SpatialIndex` |
| 2 | No R-tree | `index/rtree.py` — STR bulk-loaded R-tree behind `SpatialIndex` |
| 3 | No `eval/benchmark.py`, no `--sizes` / `--compare` | `eval/benchmark.py` — multi-size, multi-index harness |
| 4 | No end-to-end p50/p95 | `eval/bench_e2e.py` — in-process + HTTP percentile harness |
| 5 | README tables are `—` | `eval/emit_readme_tables.py` + marker-based README patching |

---

## Step 1 — Make brute force a first-class, timed competitor

**File: `index/brute.py`** (new, ~40 lines)

```python
class BruteForceIndex:  # implements SpatialIndex Protocol
    def build(self, pois: list[POI]) -> None
        # store as a single (n, 2) float64 np.ndarray of planar (x, y)
        # + parallel np.ndarray of ids. NOT a Python list of POI objects.
    def range_query(self, x, y, radius_m) -> list[int]      # vectorized:
        # mask = (xs - x)**2 + (ys - y)**2 <= radius_m**2 ; return ids[mask]
    def k_nearest(self, x, y, k) -> list[int]
        # np.argpartition on squared distances, then sort the k winners
```

Design decisions to hold:
- **Vectorized NumPy, not a Python loop.** The README's speedup column is only honest if brute force is a *competent* baseline. Beating a pure-Python loop is beating a strawman; beating vectorized NumPy is the real claim. State this in the README's design notes.
- Return **ids** (ints), not POI objects, from all indices — hydration is the store's job and must not pollute index timings.
- Keep the existing `brute_force_range` oracle function untouched in `eval/oracle.py`; `BruteForceIndex` is a competitor, the oracle stays the referee. They should share one distance predicate to avoid drift (import it from one place, e.g. `index/_dist.py`).

**Acceptance:** oracle set-equality vs. `KDTreeIndex` on ≥50 random (origin, radius) pairs at n=120k; `pytest tests/test_brute_index.py` green.

---

## Step 2 — Implement the R-tree (or the README column dies)

Decision: **implement it.** It's a preferred-qualification talking point (production spatial DBs are R-tree family) and the compare harness makes the k-d vs. R-tree tradeoff measurable instead of asserted.

**File: `index/rtree.py`** (new, ~180–250 lines)

```python
@dataclass
class _Node:
    bbox: tuple[float, float, float, float]   # (minx, miny, maxx, maxy)
    children: list["_Node"] | None            # internal
    entries: np.ndarray | None                # leaf: rows of (x, y, id)

class RTreeIndex:  # implements SpatialIndex
    def __init__(self, leaf_capacity: int = 32, fanout: int = 16): ...
    def build(self, pois) -> None             # STR bulk load (below)
    def range_query(self, x, y, radius_m) -> list[int]
    def k_nearest(self, x, y, k) -> list[int]
```

**Build = STR (Sort-Tile-Recursive) bulk loading, not incremental insert.**
1. Compute planar points once via `core.coords.to_xy` (the funnel — no direct projection calls).
2. Sort by x; slice into `ceil(sqrt(n/leaf_capacity))` vertical strips.
3. Within each strip sort by y; cut into leaves of `leaf_capacity`.
4. Compute leaf bboxes; recurse the same tiling on bbox centers with `fanout` until one root.

Why STR and how to defend it: your dataset is static-at-startup (DB is truth, index is derived, rebuilt on boot), so incremental insert/split machinery (quadratic split, R\*) is complexity with no payoff. STR gives near-optimal packing in O(n log n) build. *If asked about dynamic inserts in an interview: that's when linear/quadratic split or R\* variants earn their keep.*

**`range_query` (circle query against bbox tree):**
- Descend only into nodes whose bbox intersects the circle: test = clamp circle center to bbox, check squared distance ≤ r². (Bbox-circle, not bbox-bbox — a circle's bounding square over-admits corner nodes.)
- At leaves, exact vectorized distance test on the entry array.

**`k_nearest`:** best-first search with a heap keyed on `mindist(query, bbox)` for nodes and exact distance for points; pop until k points popped. This is the textbook (Hjaltason–Samet) algorithm; name it in comments.

Correctness traps to test explicitly (`tests/test_rtree.py`):
- duplicate points; all points collinear (degenerate strips); n < leaf_capacity (root is a leaf); query circle entirely outside all bboxes; query point exactly on a bbox edge; radius 0.
- property test: for 200 random queries, `RTreeIndex.range_query` set-equals the oracle at n ∈ {100, 5k, 120k}.

**Acceptance:** same oracle gate as Step 1, plus build time at n=120k recorded (expect well under 1 s; if not, the strip math is wrong).

---

## Step 3 — `eval/benchmark.py`: the multi-size, multi-index harness

**New file.** The CLI the README promises, exactly:

```bash
python eval/benchmark.py --sizes 1000 10000 100000 \
    --compare kdtree brute_force rtree \
    --queries 50 --reps 30 --warmup 5 --radius-m 800 --seed 42 \
    --out eval/results/index_compare.json
```

**Structure:**

```python
INDEXES = {"kdtree": KDTreeIndex, "brute_force": BruteForceIndex, "rtree": RTreeIndex}

def run(sizes, names, queries, reps, warmup, radius_m, seed) -> dict:
    for n in sizes:
        pois = generate_pois(n, seed)               # same seed → same data for all indices
        origins = sample_origins(pois, queries, seed)
        # 1. CORRECTNESS GATE (before any timing):
        #    every index's result set must equal the oracle on every origin.
        #    On mismatch: abort the whole run, print the diff. Never time wrong code.
        # 2. TIMING:
        #    per index: build once; per rep, run ALL `queries` origins, record total/queries.
        #    discard `warmup` reps; keep `reps` measurements.
        # 3. Record build time separately from query time.
```

**Protocol rules (carry over from the verification methodology):**
- Warm-up reps discarded; **median + IQR** per cell, never a single mean — absolutes on a shared machine drift ±30%, and the harness must not pretend otherwise.
- One rep = the *batch* of 50 queries (per-query µs derived by division). Timing single sub-ms queries individually measures the timer, not the index.
- `time.perf_counter_ns`; no I/O, no hydration, no printing inside the timed region.
- Capture environment in the JSON: python version, platform, CPU count, timestamp, git SHA, all CLI args. A benchmark number without its environment is a rumor.
- **Self-consistency gate:** if any index examines <1% of candidates yet shows <5× speedup vs. brute force at n=100k, print a warning pointing at the stage decomposition (`bench_stages.py`) — that's the known Amdahl signature, and this harness measures *index stage only*, so here it usually means a harness bug (e.g., hydration leaked into the timed region).

**Output JSON shape (the contract Step 5 consumes):**
```json
{
  "env": {...},
  "config": {...},
  "results": [
    {"n": 100000, "index": "kdtree",
     "build_ms": {"median": ..., "iqr": [..., ...]},
     "query_us_per_query": {"median": ..., "iqr": [..., ...]},
     "candidates_examined_frac": 0.0052}
  ]
}
```

**Also:** delete or alias the README's old command — keep `bench_stages.py` for *stage* decomposition (its job is different: where time goes inside one pipeline), and say so in a comment header in both files so nobody "consolidates" them into one confused script later.

**Acceptance:** running the exact README command produces the JSON + a printed markdown table; rerunning with the same seed reproduces medians within IQR.

---

## Step 4 — `eval/bench_e2e.py`: end-to-end p50/p95

Two modes, because "end-to-end" is ambiguous and the README table needs one number with a footnote:

**Mode A — in-process (default, deterministic):** time `reach.pipeline.reachable_pois(...)` directly.
- Workload: ≥300 requests over a **Zipf-distributed** origin pool (say 20 origins, exponent ~1.2) — uniform origins would understate cache benefit; all-one-origin overstates it. State the distribution in the output.
- Report **cold and warm separately**: run once with `ring_cache` disabled (cold p50/p95) and once enabled after a priming pass (warm p50/p95). A single blended number hides the cache, which is the whole point of the cache.
- Percentiles: `np.percentile(samples, [50, 95])` over per-request wall times; also report n, min, max. With 300+ samples, p95 has ~17+ tail samples — the minimum for the number to mean anything; don't report p99 at this sample size.

**Mode B — HTTP (`--http http://localhost:8000`):** hit `POST /search/structured` with `httpx` (sync client, single connection, keep-alive), same workload. This includes FastAPI/serde overhead and is the honest "service latency." Never mix Mode A and B numbers in one table row.

**Exclusions, stated in output:** LLM parse time is *not* in either mode (network-bound, provider-dependent; it would swamp and destabilize the geometry story). `parser_eval.py` owns LLM quality; if you want its latency, report it there as its own p50/p95 line.

CLI:
```bash
python eval/bench_e2e.py --requests 300 --origins 20 --zipf 1.2 \
    --minutes 10 [--http URL] --seed 42 --out eval/results/e2e.json
```

**Acceptance:** JSON with `{cold: {p50_ms, p95_ms, n}, warm: {...}, mode, workload, env}`; warm p50 visibly < cold p50 (if not, the cache or the workload distribution is broken — investigate before publishing).

---

## Step 5 — Fill the tables (and make refilling one command)

**File: `eval/emit_readme_tables.py`** (new, ~80 lines)

- Reads `eval/results/index_compare.json`, `e2e.json`, and `parser_eval.json`.
- Emits the two README tables as markdown, medians with IQR in parens, e.g. `0.53 (0.36–0.61)`.
- Patches `README.md` **between HTML markers** so refresh is idempotent:

```markdown
<!-- BENCH_TABLE:BEGIN (autogenerated by eval/emit_readme_tables.py — do not hand-edit) -->
| Dataset size | Brute-force (µs/q) | k-d tree (µs/q) | R-tree (µs/q) | Speedup (best vs brute) |
...
<!-- BENCH_TABLE:END -->
```

- Footnotes emitted with the table, automatically: environment one-liner (CPU, python, date, git SHA), "median (IQR), N reps × M queries, seed S", e2e mode + cold/warm, and "LLM parse excluded from e2e; see parser table."
- `--check` flag for CI: exits nonzero if the README's generated blocks don't match current JSONs (catches "code changed, README stale").

**README edits (hand-made, once):**
1. Replace the old benchmark command with the real Step 3 CLI.
2. Insert the marker pairs around both tables; delete the `—` placeholder rows.
3. Add one design note: "Brute-force baseline is vectorized NumPy, not a Python loop — see Step 1 rationale."
4. Keep the R-tree column (it exists now); add k-d vs. R-tree observed tradeoff to Design Notes *after* you have the numbers, not before.
5. Parser accuracy row: run `parser_eval.py --emit-json eval/results/parser_eval.json` (add that flag if missing) and let the emitter fill it.

**Run order to go from `—` to shipped:**
```bash
pytest tests/                          # all oracle gates green first
python eval/benchmark.py --sizes 1000 10000 100000 --compare kdtree brute_force rtree --out eval/results/index_compare.json
python eval/bench_e2e.py --requests 300 --out eval/results/e2e.json
python eval/parser_eval.py --emit-json eval/results/parser_eval.json
python eval/emit_readme_tables.py      # README now has real numbers + footnotes
git add -A && git commit -m "benchmarks: fill README tables from harness output"
```

---

## Guardrails for the whole effort

1. **Never time code that hasn't passed its oracle in the same run.** The correctness gate runs inside the benchmark, not just in CI — data or seed drift between test-time and bench-time is exactly how wrong-but-fast wins.
2. **Shares and ratios travel; absolute ms don't.** Publish speedup ratios and per-stage shares prominently; absolutes always carry the environment footnote.
3. **One distance predicate, one projection funnel.** All three indices and the oracle import the same `to_xy` and the same squared-distance helper. Two implementations of "distance" is how set-equality gates start flaking.
4. **Interview framing you buy with this work:** you'll be able to say — "my README made claims my code couldn't back; I built the harness, found the honest numbers, and wired the docs to regenerate from results so they can't drift again." That's the CI-for-claims story, and it's rarer than the algorithms themselves.
