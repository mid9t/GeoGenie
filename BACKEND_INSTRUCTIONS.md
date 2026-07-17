# GeoGenie Backend — Coding Instructions & Architecture Overview

This document specifies the backend design for GeoGenie, incorporating the findings of the Phase 2 Verification Report. Design decisions that exist *because of* a verified finding are tagged **[VR §x.y]** so you can defend each one in an interview with a measurement, not a hunch.

---

## 1. Input & Output of the System

### System input
| Input | Format | Entry point |
|---|---|---|
| Free-text user query | `{"query": "quiet accessible cafe within 10 min walk", "origin": {"lat": 37.77, "lon": -122.42}}` | `POST /search` |
| Structured query (bypass LLM) | `{"radius_m": 800, "filters": {...}, "origin": {...}}` | `POST /search/structured` |
| POI dataset | CSV/SQLite rows: `id, lon, lat, category, accessible, noise_level, hours` | ingest script, startup |
| Street graph (Phase 4) | edge list: `node_a, node_b, length_m, step_free` | ingest script |

### System output
| Output | Format |
|---|---|
| Ranked results | JSON: POIs + distance + attributes + LLM explanation |
| Parsed query echo | The structured query the LLM produced (for transparency/debugging) |
| Reachability polygon | GeoJSON ring (alpha shape) for frontend rendering |
| Benchmarks/eval | Markdown + JSON reports from the eval harness |

**Coordinate convention — fixed globally: `(lon, lat)` everywhere in storage and APIs, converted to planar `(x, y)` meters only through one function.** [VR §3.2 — the silent `(lat,lon)` swap is the highest-risk defect found; it fails without raising because `cos(−122.42°)` is arithmetically valid]

---

## 2. Project Breakdown (backend modules)

1. **`core/`** — coordinate discipline + shared types (the seam-guard layer)
2. **`geometry/`** — the verified Phase 2 package (hull, alpha shape, PIP, Voronoi, Delaunay, DP, projection). Treat as a library; do not fork it.
3. **`index/`** — Phase 1 spatial index behind an interface (k-d tree now; R-tree/PostGIS later)
4. **`store/`** — database access (POIs, street graph)
5. **`reach/`** — reachability pipeline: index prefilter → ring build → exact PIP (with caching)
6. **`routing/`** — Phase 4 accessible A*/Dijkstra
7. **`genai/`** — LLM query parsing + result explanation
8. **`api/`** — FastAPI service layer
9. **`eval/`** — oracle tests + staged benchmarking harness
10. **`optimization/`** — Phase 5 LP facility location (independent of the query path)

---

## 3. Files, Functions, and Code Types

### `core/coords.py` — pure functions, zero dependencies
```python
def to_xy(lon: float, lat: float, origin: Origin) -> tuple[float, float]
def to_lonlat(x: float, y: float, origin: Origin) -> tuple[float, float]
class Origin(NamedTuple): lon: float; lat: float
```
- **The only place** projection is called. Every other module imports `to_xy`; direct calls to `geometry.projection` outside this file should fail CI (add a lint rule / grep test). [VR §3.2 — "funnel every conversion through exactly one function"]
- **Validate `-90 ≤ lat ≤ 90` and raise** — converts silent corruption into an immediate crash. [VR §3.2 stricter fix]
- **Guard `|lat| ≤ 70°`, not 85°** — error is `tan(lat₀)·Δlat`; at 80° it's already 1.27%, the 85° guard "implies a safety it doesn't deliver." [VR §3.1 — apply the recommended-not-applied fix here, in *your* layer, without modifying the geometry package]

### `core/types.py` — dataclasses only
`POI`, `StructuredQuery`, `SearchResult`, `ReachRing` (frozen dataclasses; `StructuredQuery` also mirrored as a Pydantic schema in `api/schemas.py`).

### `index/base.py` + `index/kdtree.py`
```python
class SpatialIndex(Protocol):
    def build(self, pois: list[POI]) -> None
    def range_query(self, x, y, radius_m) -> list[POI]     # circle, planar
    def k_nearest(self, x, y, k) -> list[POI]
```
- `kdtree.py` implements it (your Phase 1 tree). The Protocol is the seam for swapping in an R-tree or PostGIS later without touching callers.
- **Do not spend further effort optimizing the tree: it is 1.8% of runtime (0.53 ms locating 629 of 120,000 POIs).** [VR §4.2–4.3]
- Caveat to carry: the report benchmarked a stand-in tree; re-run `eval/bench_stages.py` against yours once wired to confirm the 1.8% share holds. [VR §6.6]

### `store/db.py` — I/O layer
```python
def load_pois(db_path) -> list[POI]                  # startup: full load into index
def get_pois_by_ids(ids: list[int]) -> list[POI]     # hydrate attributes at rank time
def load_street_graph(db_path) -> Graph              # Phase 4
```

### `reach/pipeline.py` — the composition layer (rewrite of the unrequested VR pipeline, now requested)
```python
def reachable_pois(origin, minutes, index, walk_graph=None) -> ReachResult
    # 1. radius = minutes * WALK_SPEED_M_PER_MIN * SLACK   (crude circle)
    # 2. candidates = index.range_query(...)               (prefilter, ~0.5 ms)
    # 3. ring = ring_cache.get_or_build(origin, minutes)   (alpha shape from walk frontier)
    # 4. survivors = pip_filter(candidates, ring)          (exact test)
```
- **Alpha shape for the ring, never convex hull as the final filter** — the hull over-reported reachable POIs by 61.7% on a carved-wedge frontier (407 vs 156). Hull is admissible only as an over-approximating *prefilter* (false positives, never false negatives). [VR §3.3]
- Note honestly when discussing: 61.7% is a property of the synthetic wedge, not of any real city. [VR §3.3 caveat, §6.2]

### `reach/ring_cache.py`
```python
class RingCache:  # key: (round(origin, 4), minutes) -> ReachRing, LRU + TTL
```
- Ring build is 41% of query time and fixed per origin; "users re-query far more often than they relocate." [VR §4.3 target #2]

### `reach/pip_fast.py`
```python
def pip_filter_vectorized(points: np.ndarray, ring: np.ndarray) -> np.ndarray  # bool mask
def pip_filter_convex_fastpath(...)   # O(log n) when ring is convex
```
- The polygon test is the #1 bottleneck: 57% of runtime, ~27 µs/point in interpreted Python. Vectorize the ray cast with NumPy; keep the geometry package's scalar version as the correctness oracle in tests. [VR §4.3 target #1]

### `routing/accessible_path.py` (Phase 4)
```python
def shortest_path(graph, src, dst, require_step_free: bool) -> Route
def walk_frontier(graph, origin, minutes) -> list[tuple[float, float]]  # feeds ring build
```

### `genai/query_parser.py`
```python
def parse_query(text: str) -> StructuredQuery   # LLM w/ tool schema, temperature=0
def validate(sq: StructuredQuery) -> StructuredQuery  # clamp radius, whitelist filters
```
Failure mode: on parse failure, return HTTP 422 with the raw LLM output attached — never guess a default radius silently (same "fails silently" class of bug as VR §3.2).

### `genai/explain.py`
```python
def explain_results(query, results, excluded_sample) -> list[str]
```

### `api/main.py`, `api/schemas.py`, `api/deps.py`
- Endpoints: `POST /search`, `POST /search/structured`, `GET /reach_ring` (GeoJSON), `GET /healthz`, `GET /debug/last_parse`
- `deps.py`: builds index + cache once at startup (FastAPI lifespan), injects into handlers.

### `optimization/facility_location.py` (Phase 5)
```python
def solve_placement(demand_pts, candidate_sites, budget, dist_fn) -> Placement  # PuLP/linprog
```
Uses `core/coords.to_xy` for the distance matrix — same funnel, same guard.

### `eval/` — see §7.

---

## 4. File Structure & Interaction Map

```
geogenie/
├── core/         coords.py, types.py
├── geometry/     [pre-existing, verified 47/48 — DO NOT MODIFY]
├── index/        base.py, kdtree.py
├── store/        db.py, ingest.py
├── reach/        pipeline.py, ring_cache.py, pip_fast.py
├── routing/      graph.py, accessible_path.py
├── genai/        query_parser.py, explain.py
├── api/          main.py, schemas.py, deps.py
├── optimization/ facility_location.py
├── eval/         oracle.py, bench_stages.py, parser_eval.py, labeled_queries.json
└── tests/        test_adversarial.py [keep], test_coords_guard.py, test_pip_fast.py, ...
```

**Interaction flow (one search request):**
```
api/main.py
  └─ genai/query_parser.py ──► StructuredQuery
       └─ reach/pipeline.py
            ├─ core/coords.to_xy          (ONLY projection entry point)
            ├─ index/kdtree.range_query   (prefilter, ~2% of time)
            ├─ reach/ring_cache           (41% → amortized toward 0 on cache hit)
            │    └─ routing/walk_frontier + geometry/alpha_shape
            ├─ reach/pip_fast             (57% → the optimization target)
            └─ store/db.get_pois_by_ids   (hydrate attributes)
       └─ genai/explain.py ──► rationale strings
  └─ JSON response
```

Dependency rule: `geometry/` and `core/` import nothing from other GeoGenie modules; `reach/` imports both; `api/` imports everything; nothing imports `api/`.

---

## 5. Database Interaction

- **Engine:** SQLite for the project (single file, zero ops). Schema and access code written so Postgres+PostGIS is a config swap — that swap story is itself interview material (hand-built R-tree vs. PostGIS's production R-tree).
- **Schema:**
  ```sql
  CREATE TABLE pois (
    id INTEGER PRIMARY KEY,
    lon REAL NOT NULL, lat REAL NOT NULL,      -- (lon, lat) order, enforced by ingest
    category TEXT, accessible INTEGER,
    noise_level TEXT, hours TEXT,
    CHECK (lat BETWEEN -90 AND 90), CHECK (lon BETWEEN -180 AND 180)
  );
  CREATE TABLE graph_edges (
    node_a INTEGER, node_b INTEGER,
    length_m REAL, step_free INTEGER
  );
  ```
  The `CHECK` constraints are the database-level twin of the `core/coords` guard — the lon/lat swap gets caught at ingest, not at query time. [VR §3.2]
- **Pattern:** DB is the source of truth; the k-d tree is a **derived, in-memory structure rebuilt at startup** from `load_pois()`. No index persistence in v1 (120k points builds in well under a second). Attribute filtering happens *after* spatial filtering, via `get_pois_by_ids` — spatial index stays lean (coords + id only).

---

## 6. Frontend Interaction

Backend is a pure JSON/GeoJSON REST API — no server-side rendering. The contract:

| Endpoint | Frontend use |
|---|---|
| `POST /search` | main search box → ranked result cards (name, distance, badges, explanation) |
| `GET /reach_ring?lat=..&lon=..&minutes=..` | GeoJSON polygon → draw the reachable area on the map (Leaflet/folium) |
| `POST /search/structured` | filter-panel UI that skips the LLM |
| `GET /debug/last_parse` | dev panel showing what the LLM understood |

Rules:
- All GeoJSON follows the spec's `[lon, lat]` ordering — which conveniently matches the internal convention, so no per-endpoint flipping. [VR §3.2]
- Errors are structured: `{"error": "PARSE_FAILED", "detail": ..., "raw_llm_output": ...}` — the frontend can show "I didn't understand, did you mean…" instead of a blank failure.
- CORS enabled for the dev origin only.

---

## 7. Evaluation & Comparison Components — precise parameters

The verification report's methodology is the template: **oracle-based correctness, staged latency shares (not absolutes), honest variance reporting.** [VR §2, §4.4]

### 7.1 Correctness oracles (`eval/oracle.py`, run in CI)
| Component under test | Oracle | Pass criterion | Volume |
|---|---|---|---|
| `index.range_query` | brute-force linear scan | **exact set equality** of returned IDs | ≥ 50 random (origin, radius) pairs on 120k POIs |
| `pip_fast` (vectorized) | geometry package scalar raycast | identical bool mask | ≥ 18,000 point-ring queries incl. rays through vertices, points on edges, U-notches [VR §2 volumes] |
| `core.to_xy` distances | haversine | rel. error < 0.6% within guard (≤70° lat); **raises** beyond guard and for \|lat\|>90 | 2,000 random pairs ≤15 km at lat₀ ∈ {0°, 40°, 60°, 69°}; plus assert-raises at 71°, 80°, 91° [VR §3.1] |
| full `reachable_pois` | linear scan + scalar PIP, no cache | exact set equality | 20 origins × 3 radii |
| degeneracies | enumerated by hand, not sampled | per-case | collinear sets, duplicates, grid-snapped coords [VR §2 — "measure zero under random sampling, probability ~1 in production"] |

### 7.2 Latency benchmark (`eval/bench_stages.py`)
- **Stages timed separately:** index query / ring build (cold + cache-hit) / PIP filter / DB hydrate / LLM parse (reported separately — network-bound, would swamp everything else).
- **Protocol:** 5 warm-up reps discarded, ≥ 30 measured reps, report **median + IQR** per stage. Never a single mean. [VR §4.4 — absolutes moved 32% between runs; shares moved ~1 point]
- **Headline numbers = shares, not milliseconds.** Target after optimization: PIP share from 57% → <20% (vectorized), ring share → ~0 amortized at ≥80% cache hit rate on a Zipf-distributed origin workload.
- **Sanity gate (the §4.1 lesson):** the harness asserts internal consistency — if candidates examined is <1% of dataset but end-to-end speedup is <5×, print the stage decomposition and flag it, so a misleading "2×" never gets published unexamined.
- Scale sweep: n ∈ {1k, 10k, 100k, 500k}; verify index stage grows ~O(log n + k) while PIP stage grows with survivors only.

### 7.3 Hull vs. alpha-shape over-report (`eval/bench_stages.py --overreport`)
- Metric: `(|hull ∩ POIs| − |alpha ∩ POIs|) / |alpha ∩ POIs|` on frontiers with carved wedges of varying angle (30°–120°).
- Report as a **curve vs. wedge angle**, explicitly labeled synthetic — one number (61.7%) is a demonstration, not a calibration. [VR §3.3 caveat]
- Invariant checked alongside: hull survivors ⊇ alpha survivors always (admissibility — false positives only). [VR §3.3]

### 7.4 LLM parser eval (`eval/parser_eval.py`)
- Set: ≥ 50 hand-labeled `(text, StructuredQuery)` pairs in `labeled_queries.json`, incl. ≥10 adversarial (ambiguous units, negations like "not loud", missing origin).
- Metrics: **per-field exact match** (radius within ±10%, filters exact, sort exact) and **full-query exact match**; 3 runs at temperature 0, report worst-run numbers.
- Failure taxonomy logged: wrong-field / hallucinated-filter / refused / invalid-JSON.

### 7.5 Voronoi scope guard
- Voronoi built **only on viewport survivors** (k-d prefilter first), never dataset-wide: 14 ms on 185 sites, naive-n² extrapolation to 120k is ~95 minutes (order-of-magnitude estimate only). Enforce `n ≤ 2,000` with a raised error. [VR §4.5]

---

## 8. Desired Output at Each Stage

| Stage | Deliverable | "Done" looks like |
|---|---|---|
| A. Core seam | `core/coords.py` + guard tests | swap-order bug is now an exception; 71°/91° lat raise; CI grep proves single projection entry point |
| B. Store + ingest | SQLite with CHECK constraints, loader | 120k POIs ingest; bad-order rows rejected at ingest |
| C. Index behind Protocol | `index/kdtree.py` wired | oracle set-equality passes on 50 queries; stage share re-measured with *your* tree [VR §6.6] |
| D. Reach pipeline v1 | pipeline + alpha ring, scalar PIP | end-to-end oracle equality; stage decomposition report generated |
| E. Optimizations | `pip_fast.py` + `ring_cache.py` | PIP share <20%, cache hit ≥80% on replayed workload; before/after shares table |
| F. GenAI layer | parser + explainer + eval set | ≥90% per-field accuracy on labeled set; structured 422s on failure |
| G. API | FastAPI + GeoJSON ring endpoint | curl examples in README work; ring renders in folium |
| H. Routing (Ph. 4) | accessible A* + real walk frontier | ring built from graph frontier, not synthetic circle; step-free toggle changes routes |
| I. LP (Ph. 5) | facility location | placement solution + map visualization |
| J. Final eval | full benchmark + eval report | median+IQR tables, share charts, over-report curve, parser confusion table |

Stages A–D are the minimum demoable backend; E is where the measured wins live; F–G make it a product; H–J complete the JD story.

---

## 9. Reusable & Scalable Parts (interview prep — know these cold)

### Reusable
1. **The coordinate funnel (`core/coords.py`).** One auditable entry point for a class of bug that "doesn't raise" and makes "every downstream result confidently wrong." Transfers to any geo system; it's an interface-design answer, not a geometry answer. [VR §3.2]
2. **The oracle-based eval harness (`eval/oracle.py`).** Test against external ground truth (haversine, brute force, mathematical definitions) rather than tests written alongside the implementation — "tests written alongside an implementation encode the author's mental model." Reusable against any future index/geometry replacement, including PostGIS. [VR §2]
3. **`SpatialIndex` Protocol.** k-d tree, R-tree, PostGIS all satisfy the same 3-method interface; oracle suite validates any of them unchanged.
4. **The staged-benchmark harness** with warm-ups, medians+IQR, share-based reporting, and the self-consistency gate. Reusable for profiling any pipeline. [VR §4]

### Scalable
1. **Prefilter → exact pattern.** Cheap admissible over-approximation (index circle, hull) followed by exact check (alpha PIP, later street-graph). This is *the* production spatial-systems pattern, and you have measurements for both halves: index does its job in 0.5 ms; hull alone would be 62% wrong. [VR §3.3, §4.2]
2. **Amdahl's-law story.** "Candidates examined: 0.52%; speedup: 2×. Those can't both be innocent." Asymptotics pointed at the index; profiling proved the index was 1.8% and geometry was 98%. The scaling roadmap targets the measured bottleneck (vectorized PIP, ring cache), not the theoretically-fancy component. This is the single strongest interview narrative in the project. [VR §4.1–4.3]
3. **Cache placement chosen from workload shape** — ring build is per-origin-fixed and users re-query more than they relocate, so caching converts 41% of latency to ~0 amortized. [VR §4.3]
4. **Derived-index architecture.** DB as truth, in-memory index as disposable derived state → horizontal scaling is "run more replicas, each rebuilds at startup"; growing past memory is "swap the Protocol implementation to PostGIS."
5. **Honest limits as scalability boundaries:** projection valid to 70° lat (beyond → proper map projections); Voronoi capped at viewport scale; benchmark absolutes machine-dependent, shares portable. Stating limits precisely is a senior-engineer signal. [VR §3.1, §4.4, §4.5, §6]

---

## 10. Build Order (suggested)

A → B → C → D (demoable core, oracle-verified) → E (measured optimization win) → F → G (product) → H → I → J.

Commit the stage-decomposition benchmark output *before and after* stage E — that before/after table is the artifact you'll talk through in interviews.
