# GeoGenie

**Natural-language spatial search and accessible route discovery, powered by hand-built geometric data structures and an LLM query interface.**

GeoGenie lets a user type a request like *"find a quiet, wheelchair-accessible café within a 10-minute walk that isn't too crowded right now"* and returns ranked, explainable results. An LLM parses free-text intent into a structured spatial query; a set of geometric algorithms and spatial indices — implemented from scratch, not pulled from a geometry library — execute that query efficiently over a large point-of-interest (POI) dataset.

The project exists to explore, hands-on, how computational geometry and generative AI combine in real mapping/location systems: spatial indexing for scale, geometric algorithms for reachability and routing, and LLMs for turning fuzzy human intent into precise queries.

---

## Table of Contents

- [Why This Project](#why-this-project)
- [Architecture](#architecture)
- [Features by Phase](#features-by-phase)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Benchmarks & Evaluation](#benchmarks--evaluation)
- [Design Notes & Tradeoffs](#design-notes--tradeoffs)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why This Project

Most "AI + maps" demos wrap an LLM around an existing geocoding API and call it done. GeoGenie is deliberately built the other way around: the geometric core — spatial indices, hull/containment/routing algorithms, and an optimization layer — is implemented from first principles so that performance characteristics, correctness, and tradeoffs are understood rather than assumed. The LLM sits at the boundary, translating ambiguous human language into a query the geometric engine can execute deterministically.

This split (LLM for *understanding*, hand-built geometry for *execution*) mirrors how production spatial-AI systems are typically architected: language models are good at intent extraction but are the wrong tool for guaranteeing correct, efficient spatial computation at scale.

---

## Architecture

```
                         ┌─────────────────────────┐
   User query   ───────► │   LLM Query Parser       │
 "quiet, accessible      │  (function calling /     │
  cafe, 10 min walk"     │   structured output)     │
                         └───────────┬─────────────┘
                                     │  structured query
                                     ▼
                         ┌─────────────────────────┐
                         │   Query Planner          │
                         │  (radius, filters,       │
                         │   sort strategy)         │
                         └───────────┬─────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
        ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
        │ Spatial Index  │   │ Geometry Ops   │   │ Routing / Access.  │
        │ (k-d tree /    │   │ (convex hull,  │   │ (Dijkstra / A*     │
        │  R-tree)       │   │  point-in-poly,│   │  with step-free    │
        │                │   │  Voronoi)      │   │  constraints)      │
        └───────┬────────┘   └───────┬────────┘   └─────────┬──────────┘
                 └────────────────────┼──────────────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │  Ranking & Explanation   │
                         │  (LLM-generated          │
                         │   rationale per result)  │
                         └───────────┬─────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │  FastAPI Service Layer   │
                         └─────────────────────────┘
```

**Data flow:** free text → LLM structured query → spatial index lookup → geometric filtering/reachability → (optional) accessibility-aware routing → ranked + explained results → JSON response.

---

## Features by Phase

The project is built incrementally so each phase is independently demoable.

### Phase 1 — Spatial Indexing (the geometry core)
- **k-d tree**: built from scratch — insert, nearest-neighbor, k-NN, range query
- **R-tree** (or quadtree): built from scratch, benchmarked against the k-d tree and brute-force linear scan
- Synthetic POI dataset generator (lat/lon + attributes: category, accessibility, noise level, hours)

### Phase 2 — Classical Geometry Algorithms
- **Convex hull** (Graham scan) to compute a walkable "reachable area" polygon
- **Point-in-polygon** test to check POI membership in that reachable area
- **Douglas-Peucker line simplification** for simplifying raw GPS routes
- **Voronoi diagram** for nearest-POI catchment zones

### Phase 3 — GenAI Query Layer
- LLM function-calling pipeline that converts free text into a structured query:
  ```json
  {
    "location": {"lat": 37.7749, "lng": -122.4194},
    "radius_m": 800,
    "filters": {"accessible": true, "noise_level": "quiet"},
    "sort_by": "distance"
  }
  ```
- This structured query feeds directly into the Phase 1/2 geometric engine

### Phase 4 — Accessibility-Aware Routing
- Step-free-constrained shortest path (Dijkstra / A*) over a graph with per-edge accessibility attributes
- Route/place scoring for wheelchair accessibility
- LLM-generated natural-language explanations for why a result was included or excluded

### Phase 5 — Optimization (Linear Programming)
- Facility-location formulation: given demand points and candidate sites, choose optimal placements (e.g., accessible transit shelters) minimizing average distance under a budget constraint
- Solved with `scipy.optimize.linprog` / PuLP

### Phase 6 — Evaluation & Deployment
- Hand-labeled query set (~50 examples) for measuring LLM query-parsing accuracy
- Latency benchmarking: indexed vs. brute-force search across increasing dataset sizes
- Deployed as a FastAPI service

> **Status:** see [Roadmap](#roadmap) for which phases are complete vs. in progress.

---

## Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Core geometry | Python + NumPy | Implemented from scratch; not using `scipy.spatial.KDTree` or `shapely` for the core structures — those are used only for validation/visualization |
| LLM | Anthropic / OpenAI API | Structured output / tool use for query parsing |
| Optimization | `scipy.optimize.linprog`, PuLP | Facility-location LP |
| Serving | FastAPI | REST API layer |
| Visualization | `folium`, `matplotlib` | Maps, hulls, Voronoi diagrams |
| Validation only | `shapely`, `geopandas` | Cross-checking hand-built geometry against a trusted library |

---

## Project Structure

```
geogenie/
├── data/
│   └── generate_pois.py        # synthetic POI dataset generator
├── geometry/
│   ├── kdtree.py                # k-d tree implementation
│   ├── rtree.py                 # R-tree / quadtree implementation
│   ├── convex_hull.py           # Graham scan
│   ├── point_in_polygon.py
│   ├── voronoi.py
│   └── line_simplify.py         # Douglas-Peucker
├── routing/
│   └── accessible_path.py       # Dijkstra/A* with step-free constraints
├── optimization/
│   └── facility_location.py     # LP-based site placement
├── genai/
│   ├── query_parser.py          # LLM function-calling → structured query
│   └── explain.py               # result rationale generation
├── api/
│   └── main.py                  # FastAPI app
├── eval/
│   ├── labeled_queries.json     # hand-labeled test set
│   └── benchmark.py             # latency + accuracy benchmarking
├── notebooks/
│   └── visualizations.ipynb     # hull/Voronoi/index performance plots
├── tests/
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- An API key for your chosen LLM provider (set as an environment variable, e.g. `ANTHROPIC_API_KEY`)

### Installation
```bash
git clone https://github.com/<your-username>/geogenie.git
cd geogenie
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Generate sample data
```bash
PYTHONPATH=. python -m geogenie.store.ingest --n 100000 --out data/pois.db
```

### Run the API locally
```bash
PYTHONPATH=. uvicorn geogenie.api.main:app --reload
```

### Run benchmarks
```bash
# Index compare (fills the µs/query table)
python eval/benchmark.py --sizes 1000 10000 100000 \
    --compare kdtree brute_force rtree \
    --queries 50 --reps 30 --warmup 5 --radius-m 800 --seed 42 \
    --out eval/results/index_compare.json

# End-to-end reachability p50/p95 (cold + warm; LLM parse excluded)
python eval/bench_e2e.py --requests 300 --origins 20 --zipf 1.2 \
    --minutes 10 --seed 42 --out eval/results/e2e.json

# Parser accuracy on the hand-labeled set
python eval/parser_eval.py --emit-json eval/results/parser_eval.json

# Patch README tables from the JSON artifacts
python eval/emit_readme_tables.py
```

---

## Usage Examples

**Natural language query via API:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quiet, wheelchair-accessible cafe within a 10 minute walk"}'
```

**Response (example):**
```json
{
  "results": [
    {
      "name": "Blue Fern Coffee",
      "distance_m": 640,
      "accessible": true,
      "noise_level": "quiet",
      "explanation": "Selected for step-free entrance and low reported noise level; within your 10-minute walk radius."
    }
  ],
  "query_parsed": {
    "radius_m": 800,
    "filters": {"accessible": true, "noise_level": "quiet"}
  }
}
```

**Direct use of the spatial index (no LLM):**
```python
from geogenie.index import KDTreeIndex
from geogenie.core.coords import Origin

tree = KDTreeIndex(origin=Origin(-122.42, 37.77))
tree.build(pois)
ids = tree.k_nearest(x, y, k=5)
```

---

## Benchmarks & Evaluation

Numbers below are regenerated from `eval/results/*.json` by `eval/emit_readme_tables.py`.
Re-run the harness commands in [Getting Started](#getting-started) then `python eval/emit_readme_tables.py`.
CI can enforce freshness with `python eval/emit_readme_tables.py --check`.

<!-- BENCH_TABLE:BEGIN (autogenerated by eval/emit_readme_tables.py — do not hand-edit) -->
| Dataset size | Brute-force (µs/q) | k-d tree (µs/q) | R-tree (µs/q) | Speedup (best vs brute) |
|---|---|---|---|---|
| 1,000 | 9.49 (9.08–9.85) | 8.48 (7.82–9.14) | 18.10 (16.76–22.04) | 1.1× |
| 10,000 | 64.07 (63.57–64.90) | 34.88 (33.80–37.09) | 44.27 (43.50–49.20) | 1.8× |
| 100,000 | 617.61 (613.29–627.22) | 286.54 (282.94–293.35) | 240.13 (234.09–255.42) | 2.6× |

*Index stage only — median (IQR) µs/query; 30 reps × 50 queries, warmup 5, seed 42, radius 800.0 m. Env: Python 3.13.2, macOS-26.5.2-arm64-arm-64bit-Mach-O, 8 CPUs, git fd60653, 2026-08-27T08:46:22.932274+00:00. Brute-force baseline is vectorized NumPy, not a Python loop.*
<!-- BENCH_TABLE:END -->

<!-- METRICS_TABLE:BEGIN (autogenerated by eval/emit_readme_tables.py — do not hand-edit) -->
| Metric | Value |
|---|---|
| LLM query-parsing accuracy (hand-labeled, full-query) | 100.0% (n=51) |
| Per-field parser accuracy (radius / filters / sort) | 100% / 100% / 100% |
| End-to-end reachability latency cold p50 / p95 (in-process) | 2.27 / 4.15 ms |
| End-to-end reachability latency warm p50 / p95 (in-process) | 0.50 / 2.49 ms |

*E2E mode=in-process; workload: 300 requests over 20 Zipf(1.2) origins, 10.0 min walk. LLM parse excluded from e2e (see parser rows / `parser_eval.py`). Env: Python 3.13.2, git fd60653, 2026-08-27T08:56:31.590174+00:00.*
<!-- METRICS_TABLE:END -->

---

## Design Notes & Tradeoffs

- **Why implement k-d tree and R-tree from scratch instead of using `shapely`/`scipy.spatial`?** The goal is to demonstrate and internalize the underlying algorithms and their complexity characteristics, not just to call a library. Library implementations are used only to validate correctness of the hand-built versions.
- **Brute-force baseline is vectorized NumPy, not a Python loop.** A pure-Python scan is a strawman; the README speedup column compares tree indices against a competent vectorized full scan so the claim stays honest.
- **k-d tree vs. R-tree:** k-d trees are simpler and fast for point data with balanced dimensions; R-trees (STR bulk-loaded here — our POI set is static at startup) handle region queries and mirror what production spatial databases use. Both sit behind the same `SpatialIndex` Protocol; see the compare table above for measured tradeoffs rather than assumed ones.
- **Why an LLM for query parsing instead of a rules-based parser?** Free-text location queries are highly variable in phrasing ("quiet," "not too crowded," "close by") — an LLM with structured output handles this variability without needing to hand-write an exhaustive grammar, at the cost of needing an evaluation harness to catch parsing errors. Offline demos use a deterministic heuristic parser; accuracy is measured on `eval/labeled_queries.json`.
- **Synthetic data:** POI data is synthetically generated rather than scraped, to avoid ToS/licensing issues with real map data providers and to allow controlled testing at varying dataset sizes.

---

## Roadmap

- [ ] Phase 1 — Spatial indexing (k-d tree, R-tree)
- [ ] Phase 2 — Classical geometry algorithms (hull, point-in-polygon, Voronoi, simplification)
- [ ] Phase 3 — GenAI query parsing layer
- [ ] Phase 4 — Accessibility-aware routing
- [ ] Phase 5 — LP-based facility location optimization
- [ ] Phase 6 — Evaluation harness + FastAPI deployment

*(Check off phases as completed, and update the Benchmarks section with real results.)*

---

## License

MIT License — see `LICENSE` for details.
