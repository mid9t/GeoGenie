"""FastAPI service layer for GeoGenie."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from geogenie.api.deps import get_state, init_state
from geogenie.api.schemas import (
    ErrorResponse,
    POIResult,
    SearchRequest,
    SearchResponse,
    StructuredSearchRequest,
)
from geogenie.core.coords import Origin, to_xy
from geogenie.core.types import SearchResult, StructuredQuery
from geogenie.genai.explain import explain_results
from geogenie.genai.query_parser import ParseError, get_last_parse, parse_query
from geogenie.reach.pipeline import reachable_pois
from geogenie.routing.accessible_path import WALK_SPEED_M_PER_MIN


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_state()
    yield


app = FastAPI(title="GeoGenie", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_structured(sq: StructuredQuery) -> SearchResponse:
    state = get_state()
    origin = Origin(sq.origin_lon, sq.origin_lat)
    minutes = sq.minutes
    if minutes is None:
        minutes = sq.radius_m / WALK_SPEED_M_PER_MIN

    result = reachable_pois(
        origin,
        minutes,
        state.index,
        ring_cache=state.ring_cache,
        filters=sq.filters,
    )

    idx_origin = state.index.origin or origin
    ranked: List[SearchResult] = []
    for poi in result.pois:
        ox, oy = to_xy(origin.lon, origin.lat, idx_origin)
        px, py = to_xy(poi.lon, poi.lat, idx_origin)
        dist = math.hypot(px - ox, py - oy)
        ranked.append(SearchResult(poi=poi, distance_m=dist))
    ranked.sort(key=lambda r: r.distance_m)
    if sq.sort_by == "name":
        ranked.sort(key=lambda r: (r.poi.name or ""))

    explanations = explain_results(sq, ranked)
    results = []
    for r, expl in zip(ranked, explanations):
        results.append(
            POIResult(
                id=r.poi.id,
                lon=r.poi.lon,
                lat=r.poi.lat,
                name=r.poi.name,
                category=r.poi.category,
                accessible=r.poi.accessible,
                noise_level=r.poi.noise_level,
                distance_m=round(r.distance_m, 1),
                explanation=expl,
            )
        )
    return SearchResponse(
        results=results,
        query_parsed=asdict(sq),
        reach_ring=result.ring.as_geojson(),
    )


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    try:
        state = get_state()
        return {"ok": True, "n_pois": state.n_pois}
    except RuntimeError:
        return {"ok": False}


@app.get("/debug/last_parse")
def debug_last_parse() -> Dict[str, Any]:
    return get_last_parse()


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        sq = parse_query(req.query, req.origin.lon, req.origin.lat)
    except ParseError as exc:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="PARSE_FAILED",
                detail=str(exc),
                raw_llm_output=exc.raw_llm_output,
            ).model_dump(),
        )
    return _run_structured(sq)


@app.post("/search/structured", response_model=SearchResponse)
def search_structured(req: StructuredSearchRequest):
    sq = StructuredQuery(
        origin_lon=req.origin.lon,
        origin_lat=req.origin.lat,
        radius_m=req.radius_m,
        minutes=req.minutes,
        filters=req.filters,
        sort_by=req.sort_by,
        category=req.category,
    )
    if sq.category and "category" not in sq.filters:
        sq = StructuredQuery(
            origin_lon=sq.origin_lon,
            origin_lat=sq.origin_lat,
            radius_m=sq.radius_m,
            minutes=sq.minutes,
            filters={**sq.filters, "category": sq.category},
            sort_by=sq.sort_by,
            category=sq.category,
        )
    return _run_structured(sq)


@app.get("/reach_ring")
def reach_ring(
    lon: float = Query(...),
    lat: float = Query(...),
    minutes: float = Query(10.0, gt=0),
) -> Dict[str, Any]:
    state = get_state()
    origin = Origin(lon, lat)
    result = reachable_pois(
        origin, minutes, state.index, ring_cache=state.ring_cache
    )
    return result.ring.as_geojson()
