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
python data/generate_pois.py --count 100000 --output data/pois.csv
```

### Run the API locally
```bash
uvicorn api.main:app --reload
```

### Run benchmarks
```bash
python eval/benchmark.py --sizes 1000 10000 100000 --compare kdtree brute_force
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

**Direct use of the geometric core (no LLM):**
```python
from geometry.kdtree import KDTree

tree = KDTree(points)
nearest = tree.k_nearest(query_point, k=5)
```

---

## Benchmarks & Evaluation

*(Fill in with real numbers as phases are completed — do not ship this section with placeholders.)*

| Dataset size | Brute-force (ms) | k-d tree (ms) | R-tree (ms) | Speedup |
|---|---|---|---|---|
| 1,000 | — | — | — | — |
| 10,000 | — | — | — | — |
| 100,000 | — | — | — | — |

| Metric | Value |
|---|---|
| LLM query-parsing accuracy (hand-labeled set) | — |
| End-to-end query latency (p50 / p95) | — |

---

## Design Notes & Tradeoffs

- **Why implement k-d tree and R-tree from scratch instead of using `shapely`/`scipy.spatial`?** The goal is to demonstrate and internalize the underlying algorithms and their complexity characteristics, not just to call a library. Library implementations are used only to validate correctness of the hand-built versions.
- **k-d tree vs. R-tree:** k-d trees are simpler and fast for point data with balanced dimensions; R-trees handle bounding-box/region queries better and are closer to what production spatial databases (e.g., PostGIS) use. Both are implemented so their performance can be compared directly rather than assumed from theory.
- **Why an LLM for query parsing instead of a rules-based parser?** Free-text location queries are highly variable in phrasing ("quiet," "not too crowded," "close by") — an LLM with structured output handles this variability without needing to hand-write an exhaustive grammar, at the cost of needing an evaluation harness to catch parsing errors.
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
